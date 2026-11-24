"""Web Push store, API, worker selection, and payload tests."""

from __future__ import annotations

import json
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mltgnt.media.webchat.app import create_app  # noqa: E402
from mltgnt.media.webchat.config import WebChatMediaConfig, WebPushConfig  # noqa: E402
from mltgnt.media.webchat.push import (  # noqa: E402
    PushSubscriptionStore,
    PushWorker,
    push_enabled,
    validate_delete_body,
    validate_subscription,
)
from mltgnt.media.webchat.store import WebChatStore  # noqa: E402


class _Bridge:
    def handle_event(self, _event: object) -> None:
        return None


class FakeClock:
    def __init__(self) -> None:
        from datetime import datetime, timezone

        self.now = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def _base_config(tmp_path: Path, push: WebPushConfig | None = None, **kwargs: Any) -> WebChatMediaConfig:
    return WebChatMediaConfig(
        state_dir=tmp_path / "s",
        pending_dir=tmp_path / "p",
        events_dir=tmp_path / "e",
        store_dir=tmp_path / "w",
        push=push or WebPushConfig(),
        **kwargs,
    )


def _enabled_push(**kwargs: Any) -> WebPushConfig:
    return WebPushConfig(
        enabled=True,
        vapid_public_key="BPublicKeyExample",
        vapid_private_key="private-key-material",
        vapid_subject="mailto:ops@example.com",
        **kwargs,
    )


@pytest.fixture
def enable_pywebpush(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mltgnt.media.webchat.push._pywebpush_available", lambda: True)


def test_push_enabled_requires_all_conditions(tmp_path: Path, enable_pywebpush: None) -> None:
    assert push_enabled(_base_config(tmp_path)) is False
    cfg = _base_config(tmp_path, push=_enabled_push())
    assert push_enabled(cfg) is True
    assert push_enabled(
        _base_config(
            tmp_path,
            push=WebPushConfig(
                enabled=True,
                vapid_public_key=None,
                vapid_private_key="private-key-material",
                vapid_subject="mailto:ops@example.com",
            ),
        )
    ) is False
    bad_subject = _enabled_push()
    bad_subject = WebPushConfig(
        enabled=bad_subject.enabled,
        vapid_public_key=bad_subject.vapid_public_key,
        vapid_private_key=bad_subject.vapid_private_key,
        vapid_subject="bad",
        body_text=bad_subject.body_text,
    )
    assert push_enabled(_base_config(tmp_path, push=bad_subject)) is False


class _RecordingWorker:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1


def test_app_starts_injected_worker_only_when_push_enabled(tmp_path: Path, enable_pywebpush: None) -> None:
    disabled_worker = _RecordingWorker()
    with TestClient(create_app(_base_config(tmp_path), _Bridge(), push_worker=disabled_worker)):
        pass
    assert (disabled_worker.starts, disabled_worker.stops) == (0, 0)

    enabled_worker = _RecordingWorker()
    with TestClient(
        create_app(
            _base_config(tmp_path, push=_enabled_push()),
            _Bridge(),
            push_worker=enabled_worker,
        )
    ):
        pass
    assert (enabled_worker.starts, enabled_worker.stops) == (1, 1)


def test_push_disabled_hides_api_routes(tmp_path: Path) -> None:
    client = TestClient(create_app(_base_config(tmp_path), _Bridge()))
    assert client.get("/config").json()["push_enabled"] is False
    assert client.get("/push/vapid-public-key").status_code == 404
    assert client.post("/push/subscriptions", json={"endpoint": "https://x", "keys": {}}).status_code == 404


def test_push_api_when_enabled(tmp_path: Path, enable_pywebpush: None) -> None:
    config = _base_config(tmp_path, push=_enabled_push())
    client = TestClient(create_app(config, _Bridge()))
    assert client.get("/config").json()["push_enabled"] is True
    assert client.get("/push/vapid-public-key").json() == {"public_key": "BPublicKeyExample"}

    sub = {
        "endpoint": "https://push.example/sub/1",
        "keys": {"p256dh": "k1", "auth": "k2"},
    }
    assert client.post("/push/subscriptions", json=sub).status_code == 204
    store = PushSubscriptionStore(config.store_dir)
    assert store.active() == [sub]

    assert client.request("DELETE", "/push/subscriptions", json={"endpoint": sub["endpoint"]}).status_code == 204
    assert store.active() == []

    assert client.request("DELETE", "/push/subscriptions", json={"endpoint": sub["endpoint"]}).status_code == 204


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://insecure",
        "https://",
        "https:///path",
    ],
)
def test_subscription_validation_rejects_bad_endpoint(endpoint: str) -> None:
    with pytest.raises(ValueError, match="https"):
        validate_subscription({"endpoint": endpoint, "keys": {"p256dh": "a", "auth": "b"}})


def test_subscription_validation_rejects_empty_keys() -> None:
    with pytest.raises(ValueError):
        validate_subscription({"endpoint": "https://push.example/x", "keys": {"p256dh": "", "auth": "b"}})


def test_subscription_validation_rejects_oversized_keys() -> None:
    long_key = "x" * 1025
    with pytest.raises(ValueError, match="key too long"):
        validate_subscription(
            {
                "endpoint": "https://push.example/x",
                "keys": {"p256dh": long_key, "auth": "b"},
            }
        )


def test_subscription_validation_rejects_missing_and_oversized_fields() -> None:
    with pytest.raises(ValueError, match="keys required"):
        validate_subscription({"endpoint": "https://push.example/x"})
    with pytest.raises(ValueError, match="keys.auth required"):
        validate_subscription({"endpoint": "https://push.example/x", "keys": {"p256dh": "a"}})
    with pytest.raises(ValueError, match="key too long"):
        validate_subscription(
            {
                "endpoint": "https://push.example/x",
                "keys": {"p256dh": "a", "auth": "x" * 1025},
            }
        )
    with pytest.raises(ValueError, match="endpoint too long"):
        validate_subscription(
            {
                "endpoint": "https://push.example/" + "x" * 4096,
                "keys": {"p256dh": "a", "auth": "b"},
            }
        )


def test_api_invalid_json_returns_422(tmp_path: Path, enable_pywebpush: None) -> None:
    client = TestClient(create_app(_base_config(tmp_path, push=_enabled_push()), _Bridge()))
    assert client.post("/push/subscriptions", content=b"not-json", headers={"content-type": "application/json"}).status_code == 422


def test_api_validation_422(tmp_path: Path, enable_pywebpush: None) -> None:
    client = TestClient(create_app(_base_config(tmp_path, push=_enabled_push()), _Bridge()))
    assert client.post("/push/subscriptions", json={"endpoint": "http://bad", "keys": {"p256dh": "a", "auth": "b"}}).status_code == 422
    assert client.post("/push/subscriptions", json={"endpoint": "https://", "keys": {"p256dh": "a", "auth": "b"}}).status_code == 422
    assert client.request("DELETE", "/push/subscriptions", json={"endpoint": "http://bad"}).status_code == 422
    assert client.request("DELETE", "/push/subscriptions", json={"endpoint": "https:///x"}).status_code == 422


def test_subscription_store_recovers_from_broken_lines(tmp_path: Path) -> None:
    store = PushSubscriptionStore(tmp_path / "w")
    path = store.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"op":"upsert","endpoint":"https://a","keys":{"p256dh":"1","auth":"2"}}\n'
        "not json\n"
        '{"op":"delete","endpoint":"https://a"}\n'
        '{"op":"upsert","endpoint":"https://b","keys":{"p256dh":"3","auth":"4"}}\n',
        encoding="utf-8",
    )
    assert store.active() == [{"endpoint": "https://b", "keys": {"p256dh": "3", "auth": "4"}}]



def test_worker_sends_only_new_agent_messages(tmp_path: Path) -> None:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    store.append(message_id="old", author="bot", text="skip")
    sent: list[dict[str, str]] = []
    stop = threading.Event()
    initialized = threading.Event()
    delivered = threading.Event()
    subs = PushSubscriptionStore(tmp_path / "w")
    subs.upsert({"endpoint": "https://push.example/x", "keys": {"p256dh": "a", "auth": "b"}})

    def capture(_sub: dict[str, object], payload: dict[str, str]) -> None:
        sent.append(payload)
        delivered.set()

    worker = PushWorker(
        [("general", store)],
        subs,
        _enabled_push(),
        display_names={"bot": "Bot"},
        stop_event=stop,
        send=capture,
        poll_interval_sec=0,
    )
    original_tail_start = worker._tail_start

    def signal_tail_start(store: WebChatStore) -> tuple[date, int]:
        position = original_tail_start(store)
        initialized.set()
        return position

    worker._tail_start = signal_tail_start  # type: ignore[method-assign]
    worker.start()
    initialized.wait()
    store.append(message_id="u1", author="user", text="human")
    store.revise("u1", "update", text="changed")
    store.append(message_id="n1", author="bot", text="hello")
    delivered.wait()
    worker.stop()
    assert len(sent) == 1
    assert sent[0]["body"] == WebPushConfig().body_text
    assert sent[0]["url"].endswith("/t/n1")


def test_worker_follows_next_day(tmp_path: Path) -> None:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    sent: list[dict[str, str]] = []
    stop = threading.Event()
    initialized = threading.Event()
    delivered = threading.Event()
    subs = PushSubscriptionStore(tmp_path / "w")
    subs.upsert({"endpoint": "https://push.example/x", "keys": {"p256dh": "a", "auth": "b"}})

    def capture(_sub: dict[str, object], payload: dict[str, str]) -> None:
        sent.append(payload)
        delivered.set()

    worker = PushWorker(
        [("general", store)],
        subs,
        _enabled_push(),
        display_names={},
        stop_event=stop,
        send=capture,
        poll_interval_sec=0,
    )
    original_tail_start = worker._tail_start

    def signal_tail_start(store: WebChatStore) -> tuple[date, int]:
        position = original_tail_start(store)
        initialized.set()
        return position

    worker._tail_start = signal_tail_start  # type: ignore[method-assign]
    worker.start()
    initialized.wait()
    clock.now += timedelta(days=1)
    store.append(message_id="d2", author="bot", text="next day")
    delivered.wait()
    worker.stop()
    assert len(sent) == 1


def test_worker_removes_gone_subscription(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "w", clock=FakeClock())
    subs = PushSubscriptionStore(tmp_path / "w")
    endpoint = "https://push.example/gone"
    subs.upsert({"endpoint": endpoint, "keys": {"p256dh": "a", "auth": "b"}})

    class GoneError(Exception):
        response = type("R", (), {"status_code": 410})()

    worker = PushWorker(
        [("general", store)],
        subs,
        _enabled_push(),
        display_names={},
        stop_event=threading.Event(),
        send=lambda _sub, _payload: (_ for _ in ()).throw(GoneError()),
        poll_interval_sec=0.01,
    )
    worker._maybe_notify({"kind": "message", "author": "bot", "message_id": "m", "thread_ts": None}, "general")
    assert subs.active() == []


def test_worker_continues_after_other_send_errors(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "w", clock=FakeClock())
    subs = PushSubscriptionStore(tmp_path / "w")
    subs.upsert({"endpoint": "https://push.example/a", "keys": {"p256dh": "a", "auth": "b"}})
    subs.upsert({"endpoint": "https://push.example/b", "keys": {"p256dh": "c", "auth": "d"}})
    calls = {"n": 0}

    def flaky_send(sub: dict[str, object], _payload: dict[str, str]) -> None:
        calls["n"] += 1
        if sub["endpoint"] == "https://push.example/a":
            raise RuntimeError("temporary")

    worker = PushWorker(
        [("general", store)],
        subs,
        _enabled_push(),
        display_names={},
        stop_event=threading.Event(),
        send=flaky_send,
        poll_interval_sec=0.01,
    )
    worker._maybe_notify({"kind": "message", "author": "bot", "message_id": "m", "thread_ts": None}, "general")
    assert calls["n"] == 2


def test_validate_delete_body() -> None:
    assert validate_delete_body({"endpoint": "https://x"}) == "https://x"
    with pytest.raises(ValueError):
        validate_delete_body({"endpoint": "http://x"})
    with pytest.raises(ValueError):
        validate_delete_body({"endpoint": "https://"})


def test_push_worker_start_and_stop_are_idempotent(tmp_path: Path) -> None:
    store = WebChatStore(tmp_path / "w", clock=FakeClock())
    subs = PushSubscriptionStore(tmp_path / "w")
    stop = threading.Event()
    worker = PushWorker(
        [("general", store)],
        subs,
        _enabled_push(),
        display_names={},
        stop_event=stop,
        send=lambda _sub, _payload: None,
        poll_interval_sec=0.01,
    )
    worker.start()
    worker.start()
    worker.stop()
    worker.stop()
    assert worker._thread is not None
    assert not worker._thread.is_alive()



def test_worker_notifies_with_the_row_channel_slug(tmp_path: Path) -> None:
    """Refs sumipan/nexus#7205 AC6."""
    clock = FakeClock()
    main = WebChatStore(tmp_path / "main", clock=clock)
    side = WebChatStore(tmp_path / "side", clock=clock)
    sent: list[dict[str, str]] = []
    subs = PushSubscriptionStore(tmp_path / "main")
    subs.upsert({"endpoint": "https://push.example/x", "keys": {"p256dh": "a", "auth": "b"}})
    stop = threading.Event()
    initialized = threading.Event()
    both_channels = threading.Event()

    def capture(_sub: dict[str, object], payload: dict[str, str]) -> None:
        sent.append(payload)
        if len(sent) >= 2:
            both_channels.set()

    worker = PushWorker(
        [("main", main), ("side", side)],
        subs,
        _enabled_push(),
        display_names={},
        stop_event=stop,
        send=capture,
        poll_interval_sec=0,
    )
    original_tail_start = worker._tail_start
    initialized_channels = 0

    def signal_tail_start(store: WebChatStore) -> tuple[date, int]:
        nonlocal initialized_channels
        position = original_tail_start(store)
        initialized_channels += 1
        if initialized_channels == 2:
            initialized.set()
        return position

    worker._tail_start = signal_tail_start  # type: ignore[method-assign]
    worker.start()
    initialized.wait()
    side.append(message_id="s1", author="bot", text="side")
    main.append(message_id="m1", author="bot", text="main")
    both_channels.wait()
    worker.stop()
    urls = {p["url"] for p in sent}
    assert "/c/side/t/s1" in urls
    assert "/c/main/t/m1" in urls


def test_push_disabled_when_pywebpush_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mltgnt.media.webchat.push._pywebpush_available", lambda: False)
    assert push_enabled(_base_config(tmp_path, push=_enabled_push())) is False
