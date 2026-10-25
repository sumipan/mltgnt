"""Web Push store, API, worker selection, and payload tests."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta
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
    build_notification_payload,
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
    "body",
    [
        {"endpoint": "http://insecure", "keys": {"p256dh": "a", "auth": "b"}},
        {"endpoint": "https://ok", "keys": {"p256dh": "", "auth": "b"}},
        "not-json",
    ],
)
def test_subscription_validation_errors(body: Any) -> None:
    if isinstance(body, dict):
        with pytest.raises(ValueError):
            validate_subscription(body)
    else:
        pytest.skip("handled by HTTP layer")


def test_subscription_validation_rejects_bad_endpoint() -> None:
    with pytest.raises(ValueError, match="https"):
        validate_subscription({"endpoint": "http://x", "keys": {"p256dh": "a", "auth": "b"}})


def test_api_validation_422(tmp_path: Path, enable_pywebpush: None) -> None:
    client = TestClient(create_app(_base_config(tmp_path, push=_enabled_push()), _Bridge()))
    assert client.post("/push/subscriptions", json={"endpoint": "http://bad", "keys": {"p256dh": "a", "auth": "b"}}).status_code == 422
    assert client.request("DELETE", "/push/subscriptions", json={"endpoint": "http://bad"}).status_code == 422


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


def test_notification_payload_uses_display_name_and_fixed_body() -> None:
    body_text = WebPushConfig().body_text
    row = {"message_id": "m1", "author": "bot", "thread_ts": None}
    payload = build_notification_payload(
        row,
        channel_slug="chan",
        body_text=body_text,
        display_names={"bot": "Agent"},
    )
    assert payload == {
        "title": "Agent",
        "body": body_text,
        "url": "/c/chan/t/m1",
        "tag": "m1",
    }
    reply = {"message_id": "m2", "author": "bot", "thread_ts": "root"}
    payload2 = build_notification_payload(
        reply,
        channel_slug="chan",
        body_text=body_text,
        display_names={},
    )
    assert payload2["url"] == "/c/chan/t/root"
    assert payload2["tag"] == "root"
    assert "text" not in payload2


def test_worker_sends_only_new_agent_messages(tmp_path: Path) -> None:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    store.append(message_id="old", author="bot", text="skip")
    sent: list[dict[str, str]] = []
    stop = threading.Event()
    subs = PushSubscriptionStore(tmp_path / "w")
    subs.upsert({"endpoint": "https://push.example/x", "keys": {"p256dh": "a", "auth": "b"}})
    worker = PushWorker(
        store,
        subs,
        _enabled_push(),
        channel_slug="general",
        display_names={"bot": "Bot"},
        stop_event=stop,
        send=lambda _sub, payload: sent.append(payload),
        poll_interval_sec=0.01,
    )
    worker.start()
    time.sleep(0.05)
    store.append(message_id="u1", author="user", text="human")
    store.revise("u1", "update", text="changed")
    store.append(message_id="n1", author="bot", text="hello")
    time.sleep(0.15)
    worker.stop()
    assert len(sent) == 1
    assert sent[0]["body"] == WebPushConfig().body_text
    assert sent[0]["url"].endswith("/t/n1")


def test_worker_follows_next_day(tmp_path: Path) -> None:
    clock = FakeClock()
    store = WebChatStore(tmp_path / "w", clock=clock)
    sent: list[dict[str, str]] = []
    stop = threading.Event()
    subs = PushSubscriptionStore(tmp_path / "w")
    subs.upsert({"endpoint": "https://push.example/x", "keys": {"p256dh": "a", "auth": "b"}})
    worker = PushWorker(
        store,
        subs,
        _enabled_push(),
        channel_slug="general",
        display_names={},
        stop_event=stop,
        send=lambda _sub, payload: sent.append(payload),
        poll_interval_sec=0.01,
    )
    worker.start()
    time.sleep(0.05)
    clock.now += timedelta(days=1)
    store.append(message_id="d2", author="bot", text="next day")
    time.sleep(0.15)
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
        store,
        subs,
        _enabled_push(),
        channel_slug="general",
        display_names={},
        stop_event=threading.Event(),
        send=lambda _sub, _payload: (_ for _ in ()).throw(GoneError()),
        poll_interval_sec=0.01,
    )
    worker._maybe_notify({"kind": "message", "author": "bot", "message_id": "m", "thread_ts": None})
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
        store,
        subs,
        _enabled_push(),
        channel_slug="general",
        display_names={},
        stop_event=threading.Event(),
        send=flaky_send,
        poll_interval_sec=0.01,
    )
    worker._maybe_notify({"kind": "message", "author": "bot", "message_id": "m", "thread_ts": None})
    assert calls["n"] == 2


def test_validate_delete_body() -> None:
    assert validate_delete_body({"endpoint": "https://x"}) == "https://x"
    with pytest.raises(ValueError):
        validate_delete_body({"endpoint": "http://x"})
