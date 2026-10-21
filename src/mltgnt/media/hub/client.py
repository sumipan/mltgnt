"""HubClient (MediaClient) and urllib-based HubTransport."""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal

from mltgnt.interfaces.hub import HubAddress, HubMessage, HubTransport
from mltgnt.interfaces.media import Status
from mltgnt.interfaces.turn import Attachment
from mltgnt.media.hub.config import HubMediaConfig
from mltgnt.media.hub.guard import check

__all__ = ["HubClient", "UrllibTransport"]

_log = logging.getLogger(__name__)
_ALLOWED_EXTRA = frozenset({"to", "hop", "origin_id"})


def _address_from_extra(data: Any) -> HubAddress | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("invalid to address")
    host = data.get("host")
    if not isinstance(host, str):
        raise ValueError("invalid to address")
    persona = data.get("persona")
    if persona is not None and not isinstance(persona, str):
        raise ValueError("invalid to address")
    return HubAddress(host, persona)


def _attachments_to_body(attachments: tuple[Attachment, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for att in attachments:
        item: dict[str, Any] = {"name": att.name}
        if att.content_type is not None:
            item["content_type"] = att.content_type
        if att.uri is not None:
            item["uri"] = att.uri
        out.append(item)
    return out


def _message_to_post_body(message: HubMessage) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": message.id,
        "room": message.room,
        "sender": {"host": message.sender.host},
        "text": message.text,
        "hop": message.hop,
        "kind": message.kind,
    }
    if message.sender.persona is not None:
        body["sender"]["persona"] = message.sender.persona
    if message.thread is not None:
        body["thread"] = message.thread
    if message.to is not None:
        to_body: dict[str, Any] = {"host": message.to.host}
        if message.to.persona is not None:
            to_body["persona"] = message.to.persona
        body["to"] = to_body
    if message.origin_id is not None:
        body["origin_id"] = message.origin_id
    if message.attachments:
        body["attachments"] = _attachments_to_body(message.attachments)
    return body


class UrllibTransport:
    """HTTP transport for the hub REST API (stdlib only)."""

    def __init__(
        self,
        config: HubMediaConfig,
        *,
        timeout_sec: float = 30.0,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if not math.isfinite(timeout_sec) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive finite number")
        self._config = config
        self._timeout_sec = timeout_sec
        self._environ = environ if environ is not None else os.environ

    def _token(self) -> str | None:
        return self._environ.get(self._config.token_env)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, Any] | None:
        token = self._token()
        if not token:
            _log.warning("hub transport: missing token env %s", self._config.token_env)
            return None
        url = f"{self._config.hub_url}{path}"
        data = None
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as resp:  # noqa: S310
                raw = resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            _log.warning("hub transport: %s %s failed: %s", method, path, exc)
            return None
        if not raw:
            return resp.status, None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            _log.warning("hub transport: invalid JSON from %s: %s", path, exc)
            return None
        return resp.status, parsed

    def send(self, message: HubMessage) -> str | None:
        result = self._request(
            "POST",
            f"/rooms/{message.room}/messages",
            body=_message_to_post_body(message),
        )
        if result is None:
            return None
        status, payload = result
        if status not in (200, 201):
            return None
        if isinstance(payload, dict):
            message_id = payload.get("id")
            if isinstance(message_id, str):
                return message_id
        return None

    def revise(
        self,
        room: str,
        message_id: str,
        *,
        kind: Literal["update", "status"],
        text: str | None = None,
        status: Status | None = None,
    ) -> bool:
        body: dict[str, Any] = {"kind": kind}
        if kind == "update":
            if text is None:
                return False
            body["text"] = text
        else:
            if status is None:
                return False
            body["status"] = Status(status).value
        result = self._request(
            "PATCH",
            f"/rooms/{room}/messages/{message_id}",
            body=body,
        )
        if result is None:
            return False
        status_code, _payload = result
        return status_code == 200

    def poll(
        self,
        room: str,
        after: str | None,
        *,
        limit: int = 100,
    ) -> tuple[list[HubMessage], str | None] | None:
        path = f"/rooms/{room}/messages?limit={limit}"
        if after is not None:
            path = f"{path}&after={urllib.parse.quote(after, safe='')}"
        result = self._request("GET", path)
        if result is None:
            return None
        status_code, payload = result
        if status_code != 200 or not isinstance(payload, dict):
            return None
        raw_messages = payload.get("messages")
        cursor = payload.get("cursor")
        if not isinstance(raw_messages, list):
            return None
        if cursor is not None and not isinstance(cursor, str):
            return None
        messages: list[HubMessage] = []
        for row in raw_messages:
            if not isinstance(row, dict):
                _log.warning("hub transport: skipping non-object row in poll")
                continue
            try:
                messages.append(HubMessage.from_row(row))
            except ValueError as exc:
                _log.warning("hub transport: invalid poll row: %s", exc)
        return messages, cursor

    def guard_context(
        self,
        room: str,
        thread: str | None,
    ) -> tuple[list[Mapping[str, Any]], int] | None:
        path = f"/rooms/{room}/guard-context"
        if thread is not None:
            path = f"{path}?thread={urllib.parse.quote(thread, safe='')}"
        result = self._request("GET", path)
        if result is None:
            return None
        status_code, payload = result
        if status_code != 200 or not isinstance(payload, dict):
            return None
        rows = payload.get("thread_rows")
        count = payload.get("room_today_count")
        if not isinstance(rows, list) or not isinstance(count, int):
            return None
        typed_rows: list[Mapping[str, Any]] = [r for r in rows if isinstance(r, dict)]
        return typed_rows, count


class HubClient:
    """Post and revise messages on configured hub rooms."""

    def __init__(
        self,
        config: HubMediaConfig,
        *,
        persona: str | None,
        transport: HubTransport | None = None,
    ) -> None:
        self._config = config
        self._persona = persona
        self._transport = transport if transport is not None else UrllibTransport(config)
        self._known: dict[str, tuple[str, str]] = {}
        self._pending_id: str | None = None

    def _room_for_space(self, space: str) -> str | None:
        prefix = self._config.space_prefix
        if not space.startswith(prefix):
            _log.warning("hub client: space %s does not start with prefix %s", space, prefix)
            return None
        room = space[len(prefix) :]
        if room not in self._config.rooms:
            _log.warning("hub client: unknown room %s for space %s", room, space)
            return None
        return room

    def _parse_extra(self, extra: Mapping[str, Any] | None) -> tuple[HubAddress | None, int, str | None]:
        hop = 0
        origin_id: str | None = None
        to_addr: HubAddress | None = None
        if not extra:
            return to_addr, hop, origin_id
        unknown = set(extra.keys()) - _ALLOWED_EXTRA
        if unknown:
            raise ValueError(f"unexpected extra keys: {sorted(unknown)!r}")
        if "to" in extra:
            to_addr = _address_from_extra(extra.get("to"))
        if "hop" in extra:
            hop_val = extra["hop"]
            if not isinstance(hop_val, int) or hop_val < 0:
                raise ValueError("invalid hop")
            hop = hop_val
        if "origin_id" in extra:
            oid = extra["origin_id"]
            if oid is not None and not isinstance(oid, str):
                raise ValueError("invalid origin_id")
            origin_id = oid
        return to_addr, hop, origin_id

    def post(
        self,
        text: str,
        space: str,
        thread: str | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str | None:
        room = self._room_for_space(space)
        if room is None:
            return None
        try:
            to_addr, hop, origin_id = self._parse_extra(extra)
        except ValueError as exc:
            _log.warning("hub client: invalid extra: %s", exc)
            return None
        try:
            filtered = self._config.outbound_filter(text) if self._config.outbound_filter is not None else text
        except Exception as exc:
            _log.warning("hub client: outbound_filter failed: %s", exc)
            return None
        if filtered is None:
            return None
        if self._pending_id is None:
            self._pending_id = str(uuid.uuid4())
        message_id = self._pending_id
        sender = HubAddress(self._config.host_id, self._persona)
        message = HubMessage(
            id=message_id,
            ts=None,
            room=room,
            thread=thread,
            sender=sender,
            to=to_addr,
            text=filtered,
            attachments=(),
            hop=hop,
            origin_id=origin_id,
            kind="message",
            status=None,
        )
        try:
            ctx = self._transport.guard_context(room, thread)
        except Exception as exc:
            _log.warning("hub client: guard_context failed room=%s: %s", room, exc)
            return None
        if ctx is None:
            _log.warning("hub client: guard_context unavailable room=%s", room)
            return None
        thread_rows, room_today_count = ctx
        verdict = check(
            message,
            thread_rows,
            room_today_count,
            datetime.now(timezone.utc),
            config=self._config.guard,
        )
        if not verdict.ok:
            _log.warning("hub client: guard rejected post room=%s reason=%s", room, verdict.reason)
            return None
        try:
            sent_id = self._transport.send(message)
        except Exception as exc:
            _log.warning("hub client: send failed room=%s: %s", room, exc)
            return None
        if sent_id is None:
            return None
        self._pending_id = None
        self._known[sent_id] = (room, filtered)
        return sent_id

    def update(self, message_id: str, text: str) -> bool:
        known = self._known.get(message_id)
        if known is None:
            return False
        room, _prev = known
        try:
            return self._transport.revise(room, message_id, kind="update", text=text)
        except Exception as exc:
            _log.warning("hub client: update failed message_id=%s: %s", message_id, exc)
            return False

    def set_status(self, message_id: str, status: Status) -> bool:
        known = self._known.get(message_id)
        if known is None:
            return False
        room, _prev = known
        try:
            return self._transport.revise(room, message_id, kind="status", status=status)
        except Exception as exc:
            _log.warning("hub client: set_status failed message_id=%s: %s", message_id, exc)
            return False

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        return False

    def react(self, message_id: str, name: str) -> bool:
        return False
