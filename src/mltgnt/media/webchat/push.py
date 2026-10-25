"""Web Push subscription store, validation, and tail worker for WebChat."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mltgnt.media.webchat.config import WebChatMediaConfig, WebPushConfig
from mltgnt.media.webchat.inbound import DEFAULT_AUTHOR
from mltgnt.media.webchat.store import WebChatStore

__all__ = [
    "PushSubscriptionStore",
    "PushWorker",
    "build_notification_payload",
    "push_enabled",
    "validate_delete_body",
    "validate_subscription",
]

_log = logging.getLogger(__name__)

_ENDPOINT_MAX = 4096
_KEY_MAX = 1024


def _validate_https_endpoint(endpoint: str) -> None:
    if not endpoint.startswith("https://"):
        raise ValueError("endpoint must be https")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("endpoint must be https")


def _pywebpush_available() -> bool:
    try:
        import pywebpush  # noqa: F401
    except ImportError:
        return False
    return True


def push_enabled(config: WebChatMediaConfig) -> bool:
    """True when push is configured and ``pywebpush`` can be imported."""
    push = config.push
    if not push.enabled:
        return False
    if not push.vapid_public_key or not push.vapid_private_key:
        return False
    subject = push.vapid_subject
    if not (subject.startswith("mailto:") or subject.startswith("https://")):
        return False
    return _pywebpush_available()


def validate_subscription(body: Mapping[str, object]) -> dict[str, object]:
    """Validate a PushSubscription JSON body; raise ValueError on failure."""
    endpoint = body.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint required")
    endpoint = endpoint.strip()
    if len(endpoint) > _ENDPOINT_MAX:
        raise ValueError("endpoint too long")
    _validate_https_endpoint(endpoint)
    keys = body.get("keys")
    if not isinstance(keys, Mapping):
        raise ValueError("keys required")
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")
    if not isinstance(p256dh, str) or not p256dh.strip():
        raise ValueError("keys.p256dh required")
    if not isinstance(auth, str) or not auth.strip():
        raise ValueError("keys.auth required")
    p256dh = p256dh.strip()
    auth = auth.strip()
    if len(p256dh) > _KEY_MAX or len(auth) > _KEY_MAX:
        raise ValueError("key too long")
    return {"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}}


def validate_delete_body(body: Mapping[str, object]) -> str:
    endpoint = body.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint required")
    endpoint = endpoint.strip()
    if len(endpoint) > _ENDPOINT_MAX:
        raise ValueError("endpoint too long")
    _validate_https_endpoint(endpoint)
    return endpoint


def build_notification_payload(
    row: Mapping[str, Any],
    *,
    channel_slug: str,
    body_text: str,
    display_names: Mapping[str, str],
) -> dict[str, str]:
    author = str(row.get("author") or "")
    message_id = str(row["message_id"])
    thread_ts = row.get("thread_ts")
    tag = str(thread_ts or message_id)
    url_id = str(thread_ts or message_id)
    title = display_names.get(author, author)
    return {
        "title": title,
        "body": body_text,
        "url": f"/c/{channel_slug}/t/{url_id}",
        "tag": tag,
    }


class PushSubscriptionStore:
    """Append-only JSONL log of push subscription upserts and deletes."""

    def __init__(self, store_dir: Path) -> None:
        self._path = Path(store_dir) / "push_subscriptions.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def _append(self, row: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            f.write(line)
            f.flush()

    def upsert(self, subscription: Mapping[str, object]) -> None:
        data = validate_subscription(subscription)
        self._append({"op": "upsert", "endpoint": data["endpoint"], "keys": data["keys"]})

    def delete(self, endpoint: str) -> bool:
        had = endpoint in {s["endpoint"] for s in self.active()}
        self._append({"op": "delete", "endpoint": endpoint})
        return had

    def active(self) -> list[dict[str, object]]:
        if not self._path.is_file():
            return []
        latest: dict[str, dict[str, object]] = {}
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    _log.warning("[webchat push] skip broken subscription line")
                    continue
                if not isinstance(row, dict):
                    continue
                op = row.get("op")
                endpoint = row.get("endpoint")
                if not isinstance(endpoint, str) or not endpoint:
                    continue
                if op == "delete":
                    latest.pop(endpoint, None)
                elif op == "upsert" and isinstance(row.get("keys"), dict):
                    latest[endpoint] = {
                        "endpoint": endpoint,
                        "keys": row["keys"],
                    }
        return list(latest.values())


class PushWorker:
    """Tail the message store and send Web Push notifications for agent replies."""

    def __init__(
        self,
        store: WebChatStore,
        subscriptions: PushSubscriptionStore,
        push_config: WebPushConfig,
        *,
        channel_slug: str,
        display_names: Mapping[str, str],
        stop_event: threading.Event | None = None,
        send: Callable[[dict[str, object], dict[str, str]], None] | None = None,
        poll_interval_sec: float = 0.5,
        join_timeout_sec: float = 5.0,
    ) -> None:
        self._store = store
        self._subscriptions = subscriptions
        self._push_config = push_config
        self._channel_slug = channel_slug
        self._display_names = display_names
        self._stop = stop_event if stop_event is not None else threading.Event()
        self._send = send if send is not None else self._default_send
        self._poll_interval_sec = poll_interval_sec
        self._join_timeout_sec = join_timeout_sec
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="webchat-push", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._join_timeout_sec)
        self._started = False

    def _default_send(self, subscription: dict[str, object], payload: dict[str, str]) -> None:
        from pywebpush import WebPushException, webpush

        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=self._push_config.vapid_private_key,
            vapid_claims={"sub": self._push_config.vapid_subject},
        )

    def _run(self) -> None:
        try:
            day = self._store.today()
            path = self._store.path_for(day)
            offset = path.stat().st_size if path.is_file() else 0
            while not self._stop.is_set():
                today = self._store.today()
                if today > day:
                    day = today
                    offset = 0
                rows, offset = self._store.read_from(day, offset)
                for _end, row in rows:
                    if self._stop.is_set():
                        break
                    self._maybe_notify(row)
                if self._stop.wait(self._poll_interval_sec):
                    break
        except Exception:
            _log.exception("[webchat push] worker stopped after error")

    def _maybe_notify(self, row: Mapping[str, Any]) -> None:
        if row.get("kind") != "message":
            return
        if row.get("author") == DEFAULT_AUTHOR:
            return
        payload = build_notification_payload(
            row,
            channel_slug=self._channel_slug,
            body_text=self._push_config.body_text,
            display_names=self._display_names,
        )
        for subscription in self._subscriptions.active():
            try:
                self._send(subscription, payload)
            except Exception as exc:
                code = None
                try:
                    from pywebpush import WebPushException

                    if isinstance(exc, WebPushException) and exc.response is not None:
                        code = exc.response.status_code
                except ImportError:
                    pass
                if code is None:
                    status = getattr(exc, "response", None)
                    code = getattr(status, "status_code", None) if status is not None else None
                if code in (404, 410):
                    endpoint = str(subscription.get("endpoint", ""))
                    if endpoint:
                        self._subscriptions.delete(endpoint)
                    continue
                _log.warning(
                    "[webchat push] send failed endpoint=%s exc=%s",
                    subscription.get("endpoint"),
                    type(exc).__name__,
                )
