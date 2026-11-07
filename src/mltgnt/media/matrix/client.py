"""Matrix MediaClient and urllib transport."""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from mltgnt.interfaces.media import MediaClient, Status
from mltgnt.media.matrix.config import MatrixMediaConfig
from mltgnt.media.matrix.markdown import to_html

__all__ = [
    "MatrixClient",
    "MatrixResponse",
    "MatrixTransport",
    "UrllibMatrixTransport",
]

_log = logging.getLogger(__name__)

_EMOJI: dict[str, str] = {
    "thumbsup": "\U0001f44d",
    "+1": "\U0001f44d",
    "eyes": "\U0001f440",
    "bulb": "\U0001f4a1",
    "thinking_face": "\U0001f914",
    "white_check_mark": "\u2705",
    "x": "\u274c",
    "hourglass_flowing_sand": "\u23f3",
    "warning": "\u26a0\ufe0f",
    "tada": "\U0001f389",
    "pray": "\U0001f64f",
}

_WORKING_STATUSES = frozenset({Status.RECEIVED, Status.WORKING})


@dataclass(frozen=True)
class MatrixResponse:
    status: int
    json: dict[str, Any] | None
    data: bytes = b""


@runtime_checkable
class MatrixTransport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | bytes | None = None,
        content_type: str | None = None,
        timeout_sec: float = 30.0,
    ) -> MatrixResponse: ...


def _quote_path_segment(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


class UrllibMatrixTransport:
    """HTTP transport for Matrix Client-Server API (stdlib only)."""

    def __init__(
        self,
        config: MatrixMediaConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        self._environ = environ if environ is not None else os.environ

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | bytes | None = None,
        content_type: str | None = None,
        timeout_sec: float = 30.0,
    ) -> MatrixResponse:
        token = self._environ.get(self._config.access_token_env, "")
        if not token:
            _log.warning("matrix transport: missing access token env %s", self._config.access_token_env)
            return MatrixResponse(0, None)
        url = self._config.homeserver_url + path
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        data: bytes | None = None
        headers = {"Authorization": f"Bearer {token}"}
        if isinstance(body, Mapping):
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = content_type or "application/json"
        elif isinstance(body, bytes):
            data = body
            if content_type:
                headers["Content-Type"] = content_type
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
                parsed: dict[str, Any] | None = None
                if raw:
                    try:
                        loaded = json.loads(raw.decode("utf-8"))
                        if isinstance(loaded, dict):
                            parsed = loaded
                    except json.JSONDecodeError:
                        pass
                return MatrixResponse(resp.status, parsed, data=raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            parsed: dict[str, Any] | None = None
            if raw:
                try:
                    loaded = json.loads(raw.decode("utf-8"))
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    pass
            return MatrixResponse(exc.code, parsed, data=raw)
        except OSError as exc:
            _log.warning("matrix transport: request failed: %s", exc)
            return MatrixResponse(0, None)


class MatrixClient:
    """MediaClient over the Matrix Client-Server API."""

    def __init__(
        self,
        config: MatrixMediaConfig,
        *,
        transport: MatrixTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self._config = config
        self._transport = transport if transport is not None else UrllibMatrixTransport(config)
        self._sleep = sleep if sleep is not None else time.sleep
        self._event_rooms: dict[str, str] = {}
        self._thread_last_event: dict[str, str] = {}

    def remember(self, event_id: str, room_id: str) -> None:
        self._event_rooms[event_id] = room_id

    def _room_for_message(self, message_id: str) -> str | None:
        return self._event_rooms.get(message_id)

    def _send_message(
        self,
        room_id: str,
        payload: dict[str, Any],
        *,
        txn_id: str | None = None,
    ) -> str | None:
        txn = txn_id or f"mltgnt-{uuid.uuid4().hex}"
        room = _quote_path_segment(room_id)
        path = f"/_matrix/client/v3/rooms/{room}/send/m.room.message/{_quote_path_segment(txn)}"
        resp = self._request_with_retry("PUT", path, body=payload)
        if resp is None or resp.json is None:
            return None
        event_id = resp.json.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            _log.warning("matrix client: send missing event_id room=%s", room_id)
            return None
        self._event_rooms[event_id] = room_id
        thread_id = None
        relates = payload.get("m.relates_to")
        if isinstance(relates, Mapping) and relates.get("rel_type") == "m.thread":
            thread_id = relates.get("event_id")
        if isinstance(thread_id, str):
            self._thread_last_event[thread_id] = event_id
        return event_id

    def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        body: Mapping[str, Any] | bytes | None = None,
        content_type: str | None = None,
    ) -> MatrixResponse | None:
        resp = self._transport.request(method, path, query=query, body=body, content_type=content_type)
        if resp.status == 429:
            retry_ms = 1000
            if resp.json and isinstance(resp.json.get("retry_after_ms"), int):
                retry_ms = resp.json["retry_after_ms"]
            delay = min(retry_ms, 30000) / 1000.0
            self._sleep(delay)
            resp = self._transport.request(method, path, query=query, body=body, content_type=content_type)
            if resp.status == 429:
                _log.warning("matrix client: rate limited twice on %s", path)
                return None
        if resp.status < 200 or resp.status >= 300:
            _log.warning("matrix client: %s %s -> %s", method, path, resp.status)
            return None
        return resp

    def _message_body(self, text: str) -> dict[str, Any]:
        return {
            "msgtype": "m.text",
            "body": text,
            "format": "org.matrix.custom.html",
            "formatted_body": to_html(text),
        }

    def _thread_relates(self, thread: str) -> dict[str, Any]:
        reply_to = self._thread_last_event.get(thread, thread)
        return {
            "rel_type": "m.thread",
            "event_id": thread,
            "is_falling_back": True,
            "m.in_reply_to": {"event_id": reply_to},
        }

    def post(
        self,
        text: str,
        space: str,
        thread: str | None = None,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> str | None:
        _ = extra
        room_id = self._config.room_for_space(space)
        if room_id is None:
            return None
        payload = self._message_body(text)
        if thread is not None:
            payload["m.relates_to"] = self._thread_relates(thread)
        return self._send_message(room_id, payload)

    def update(self, message_id: str, text: str) -> bool:
        room_id = self._room_for_message(message_id)
        if room_id is None:
            return False
        new_content = self._message_body(text)
        payload = {
            "msgtype": "m.text",
            "body": "* " + text,
            "m.new_content": new_content,
            "m.relates_to": {"rel_type": "m.replace", "event_id": message_id},
        }
        event_id = self._send_message(room_id, payload)
        return event_id is not None

    def set_status(self, message_id: str, status: Status) -> bool:
        room_id = self._room_for_message(message_id)
        if room_id is None:
            return False
        user = _quote_path_segment(self._config.user_id)
        room = _quote_path_segment(room_id)
        path = f"/_matrix/client/v3/rooms/{room}/typing/{user}"
        body: dict[str, Any] = {"typing": False}
        if status in _WORKING_STATUSES:
            body = {"typing": True, "timeout": 30000}
        resp = self._request_with_retry("PUT", path, body=body)
        return resp is not None

    def react(self, message_id: str, name: str) -> bool:
        emoji = _EMOJI.get(name)
        if emoji is None:
            return False
        room_id = self._room_for_message(message_id)
        if room_id is None:
            return False
        payload = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": message_id,
                "key": emoji,
            },
        }
        txn = f"mltgnt-{uuid.uuid4().hex}"
        room = _quote_path_segment(room_id)
        path = f"/_matrix/client/v3/rooms/{room}/send/m.reaction/{_quote_path_segment(txn)}"
        resp = self._request_with_retry("PUT", path, body=payload)
        return resp is not None

    def upload(self, path: str, space: str, thread: str | None = None) -> bool:
        room_id = self._config.room_for_space(space)
        if room_id is None:
            return False
        file_path = Path(path)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            _log.warning("matrix client: upload read failed: %s", exc)
            return False
        if len(data) > self._config.max_upload_bytes:
            return False
        content_type, _ = mimetypes.guess_type(str(file_path))
        if not content_type:
            content_type = "application/octet-stream"
        query = {"filename": file_path.name}
        resp = self._request_with_retry(
            "POST",
            "/_matrix/media/v3/upload",
            query=query,
            body=data,
            content_type=content_type,
        )
        if resp is None or resp.json is None:
            return False
        content_uri = resp.json.get("content_uri")
        if not isinstance(content_uri, str):
            return False
        msgtype = "m.image" if content_type.startswith("image/") else "m.file"
        body_text = file_path.name
        payload: dict[str, Any] = {
            "msgtype": msgtype,
            "body": body_text,
            "url": content_uri,
            "info": {"mimetype": content_type, "size": len(data)},
        }
        if msgtype == "m.image":
            payload["filename"] = file_path.name
        if thread is not None:
            payload["m.relates_to"] = self._thread_relates(thread)
        event_id = self._send_message(room_id, payload)
        return event_id is not None
