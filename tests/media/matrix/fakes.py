"""Fake Matrix homeserver for tests."""

from __future__ import annotations

import json
import re
import urllib.parse
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from mltgnt.media.matrix.client import MatrixResponse, MatrixTransport

__all__ = ["FakeHomeserver", "Call"]


@dataclass
class Call:
    method: str
    path: str
    query: Mapping[str, str] | None
    body: Mapping[str, Any] | bytes | None


@dataclass
class FakeHomeserver:
    """Records requests and returns queued MatrixResponse values per route key."""

    token: str = "test-token"
    calls: list[Call] = field(default_factory=list)
    _queues: dict[str, deque[MatrixResponse]] = field(default_factory=lambda: defaultdict(deque))

    def queue(self, method: str, path_pattern: str, *responses: MatrixResponse) -> None:
        key = f"{method.upper()}:{path_pattern}"
        for resp in responses:
            self._queues[key].append(resp)

    def _match_key(self, method: str, path: str) -> str | None:
        for key in self._queues:
            m, pattern = key.split(":", 1)
            if m != method.upper():
                continue
            if pattern in path or re.search(pattern, path):
                return key
        return None

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
        _ = content_type, timeout_sec
        parsed_body: Mapping[str, Any] | bytes | None = body
        if isinstance(body, bytes):
            parsed_body = body
        self.calls.append(Call(method=method, path=path, query=dict(query) if query else None, body=parsed_body))
        key = self._match_key(method, path)
        if key and self._queues[key]:
            return self._queues[key].popleft()
        if method.upper() == "GET" and "/sync" in path:
            return MatrixResponse(200, {"next_batch": "s0", "rooms": {}})
        if method.upper() == "PUT" and "/send/m.room.message/" in path:
            txn = path.rsplit("/", 1)[-1]
            return MatrixResponse(200, {"event_id": f"$sent-{txn[:8]}"})
        if method.upper() == "PUT" and "/send/m.room.reaction/" in path:
            return MatrixResponse(200, {"event_id": "$reaction"})
        if method.upper() == "PUT" and "/typing/" in path:
            return MatrixResponse(200, {})
        if method.upper() == "POST" and "/upload" in path:
            return MatrixResponse(200, {"content_uri": "mxc://x/y"})
        return MatrixResponse(404, None)

    def calls_matching(self, substring: str) -> list[Call]:
        return [c for c in self.calls if substring in c.path]

    def send_bodies(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for call in self.calls:
            if call.method.upper() != "PUT" or "/send/m.room.message/" not in call.path:
                continue
            if isinstance(call.body, Mapping):
                out.append(dict(call.body))
        return out
