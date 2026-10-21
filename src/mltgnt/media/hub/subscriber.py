"""Poll hub rooms and deliver inbound messages to a host handler."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mltgnt.interfaces.hub import HubMessage, HubTransport
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.hub.client import UrllibTransport
from mltgnt.media.hub.config import HubMediaConfig
from mltgnt.media.hub.inbound import to_media_event

__all__ = ["HubSubscriber"]

_log = logging.getLogger(__name__)
_RECENT_ID_CAP = 200
_BACKOFF_SCHEDULE = (5.0, 15.0, 60.0)


@runtime_checkable
class EventHandler(Protocol):
    def handle_event(self, event: MediaEvent) -> Any: ...


class HubSubscriber:
    """DaemonComponent that polls hub rooms and calls ``handle_event``."""

    def __init__(
        self,
        config: HubMediaConfig,
        handler: EventHandler,
        *,
        transport: HubTransport | None = None,
    ) -> None:
        self._config = config
        self._handler = handler
        self._transport = transport if transport is not None else UrllibTransport(config)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._backoff_index = 0

    @property
    def name(self) -> str:
        return "hub_subscriber"

    def start(self) -> None:
        if self._started and self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="hub-subscriber", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._config.poll_interval_sec, 60.0) + 2.0)
            self._thread = None
        self._started = False

    def _cursor_path(self, room: str) -> Path:
        return self._config.resolved_cursor_dir / f"{room}.json"

    def _load_state(self, room: str) -> tuple[str | None, list[str]]:
        path = self._cursor_path(room)
        if not path.is_file():
            return None, []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("hub subscriber: corrupt cursor for room=%s: %s", room, exc)
            corrupt = path.with_suffix(path.suffix + ".corrupt")
            try:
                path.replace(corrupt)
            except OSError:
                pass
            return None, []
        if not isinstance(data, dict):
            return None, []
        cursor = data.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            cursor = None
        recent = data.get("recent_ids")
        if not isinstance(recent, list):
            recent = []
        recent_ids = [rid for rid in recent if isinstance(rid, str)]
        return cursor, recent_ids

    def _save_state(self, room: str, cursor: str | None, recent_ids: list[str]) -> None:
        trimmed = recent_ids[-_RECENT_ID_CAP:]
        payload = {"cursor": cursor, "recent_ids": trimmed}
        directory = self._config.resolved_cursor_dir
        directory.mkdir(parents=True, exist_ok=True)
        target = self._cursor_path(room)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except OSError as exc:
            _log.warning("hub subscriber: failed to save cursor room=%s: %s", room, exc)
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def _should_deliver(self, message: HubMessage) -> bool:
        if message.kind != "message":
            return False
        if message.sender.host == self._config.host_id:
            return False
        return message.to is None or message.to.host == self._config.host_id

    def _record_recent(self, recent_ids: list[str], message_id: str) -> list[str]:
        if message_id in recent_ids:
            return recent_ids
        recent_ids.append(message_id)
        if len(recent_ids) > _RECENT_ID_CAP:
            recent_ids = recent_ids[-_RECENT_ID_CAP:]
        return recent_ids

    def _process_room(self, room: str) -> bool:
        """Poll one room. Return False when transport poll failed."""
        cursor, recent_ids = self._load_state(room)
        result = self._transport.poll(room, cursor, limit=100)
        if result is None:
            return False
        messages, batch_cursor = result
        for message in messages:
            if message.id in recent_ids:
                continue
            if self._should_deliver(message):
                event = to_media_event(message, space_prefix=self._config.space_prefix)
                try:
                    self._handler.handle_event(event)
                except Exception:
                    _log.exception(
                        "hub subscriber: handle_event failed room=%s message_id=%s",
                        room,
                        message.id,
                    )
            recent_ids = self._record_recent(recent_ids, message.id)
        if batch_cursor is not None:
            self._save_state(room, batch_cursor, recent_ids)
        return True

    def _interruptible_sleep(self, seconds: float) -> bool:
        """Wait up to ``seconds``. Return True when stop was requested."""
        return self._stop.wait(seconds)

    def _run(self) -> None:
        while not self._stop.is_set():
            poll_failed = False
            for room in self._config.rooms:
                if self._stop.is_set():
                    return
                if not self._process_room(room):
                    poll_failed = True
                    break
            if poll_failed:
                delay = _BACKOFF_SCHEDULE[min(self._backoff_index, len(_BACKOFF_SCHEDULE) - 1)]
                self._backoff_index = min(self._backoff_index + 1, len(_BACKOFF_SCHEDULE) - 1)
                if self._interruptible_sleep(delay):
                    return
                continue
            self._backoff_index = 0
            if self._interruptible_sleep(self._config.poll_interval_sec):
                return
