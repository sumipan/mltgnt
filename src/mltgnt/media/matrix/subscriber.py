"""Poll Matrix /sync and deliver inbound messages."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.matrix.client import MatrixClient, MatrixTransport, UrllibMatrixTransport
from mltgnt.media.matrix.config import MatrixMediaConfig
from mltgnt.media.matrix.inbound import to_media_event

__all__ = ["MatrixSubscriber"]

_log = logging.getLogger(__name__)
_BACKOFF_SCHEDULE = (5.0, 15.0, 60.0)
_SYNC_STATE = "matrix_sync.json"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@runtime_checkable
class EventHandler(Protocol):
    def handle_event(self, event: MediaEvent) -> Any: ...


def _mxc_parts(uri: str) -> tuple[str, str] | None:
    if not uri.startswith("mxc://"):
        return None
    rest = uri[len("mxc://") :]
    server, _, media_id = rest.partition("/")
    if not server or not media_id:
        return None
    return server, media_id


class MatrixSubscriber:
    """DaemonComponent that long-polls ``/sync`` and calls ``handle_event``."""

    def __init__(
        self,
        config: MatrixMediaConfig,
        handler: EventHandler,
        *,
        client: MatrixClient,
        transport: MatrixTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        import time

        self._config = config
        self._handler = handler
        self._client = client
        self._transport = transport if transport is not None else UrllibMatrixTransport(config)
        self._sleep = sleep if sleep is not None else time.sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._backoff_index = 0

    @property
    def name(self) -> str:
        return "matrix_subscriber"

    def start(self) -> None:
        if self._started and self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._backoff_index = 0
        self._thread = threading.Thread(target=self._run, name="matrix-subscriber", daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._config.poll_timeout_ms / 1000.0, 60.0) + 2.0)
            self._thread = None
        self._started = False

    def _state_path(self) -> Path:
        return Path(self._config.state_dir) / _SYNC_STATE

    def _load_state(self) -> tuple[str | None, list[str]]:
        path = self._state_path()
        if not path.is_file():
            return None, []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state must be an object")
            batch = data.get("next_batch")
            if batch is not None and not isinstance(batch, str):
                raise ValueError("next_batch must be a string or null")
            notified = data.get("encrypted_notified")
            if not isinstance(notified, list) or not all(isinstance(x, str) for x in notified):
                raise ValueError("encrypted_notified must be a list of strings")
            return batch, list(notified)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _log.warning("matrix subscriber: corrupt sync state: %s", exc)
            return None, []

    def _save_state(self, next_batch: str | None, encrypted_notified: list[str]) -> None:
        payload = {"next_batch": next_batch, "encrypted_notified": encrypted_notified}
        directory = Path(self._config.state_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = self._state_path()
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        except OSError as exc:
            _log.warning("matrix subscriber: failed to save sync state: %s", exc)
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def _download_file(self, url: str, info: dict[str, Any]) -> Attachment | None:
        size = info.get("size")
        if isinstance(size, int) and size > self._config.max_upload_bytes:
            return None
        parts = _mxc_parts(url)
        if parts is None:
            return None
        server, media_id = parts
        q_server = urllib.parse.quote(server, safe="")
        q_media = urllib.parse.quote(media_id, safe="")
        paths = [
            f"/_matrix/client/v1/media/download/{q_server}/{q_media}",
            f"/_matrix/media/v3/download/{q_server}/{q_media}",
        ]
        raw: bytes | None = None
        for idx, path in enumerate(paths):
            resp = self._transport.request("GET", path)
            if resp.status == 200:
                raw = resp.data
                break
            if resp.status == 404 and idx == 0:
                continue
            return None
        if raw is None or len(raw) > self._config.max_upload_bytes:
            return None
        name = str(info.get("filename") or media_id)
        safe = _SAFE_NAME.sub("_", name).strip("._") or "file"
        dest_dir = self._config.resolved_files_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{media_id}-{safe}"
        try:
            dest.write_bytes(raw)
        except OSError as exc:
            _log.warning("matrix subscriber: file write failed: %s", exc)
            return None
        mimetype = info.get("mimetype")
        ct = mimetype if isinstance(mimetype, str) else None
        return Attachment(name=name, content_type=ct, uri=str(dest))

    def _room_encrypted(self, room_block: dict[str, Any]) -> bool:
        for section in ("state", "timeline"):
            block = room_block.get(section)
            events = block.get("events") if isinstance(block, dict) else None
            if not isinstance(events, list):
                continue
            for ev in events:
                if isinstance(ev, dict) and ev.get("type") in ("m.room.encryption", "m.room.encrypted"):
                    return True
        return False

    def _notify_encrypted(self, room_id: str, encrypted_notified: list[str]) -> None:
        if room_id in encrypted_notified:
            return
        space = self._config.space_for_room(room_id)
        if space is None:
            return
        self._client.post(self._config.encrypted_room_notice, space)
        encrypted_notified.append(room_id)

    def _process_invites(self, rooms: dict[str, Any]) -> None:
        invite = rooms.get("invite")
        if not isinstance(invite, dict):
            return
        for room_id, block in invite.items():
            if room_id not in self._config.rooms.values():
                continue
            if not isinstance(block, dict):
                continue
            invite_state = block.get("invite_state")
            if not isinstance(invite_state, dict):
                continue
            events = invite_state.get("events")
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if ev.get("type") != "m.room.member":
                    continue
                if ev.get("state_key") != self._config.user_id:
                    continue
                content = ev.get("content")
                if not isinstance(content, dict) or content.get("membership") != "invite":
                    continue
                sender = ev.get("sender")
                if not isinstance(sender, str) or sender not in self._config.allowed_senders:
                    continue
                room = urllib.parse.quote(room_id, safe="")
                self._transport.request("POST", f"/_matrix/client/v3/rooms/{room}/join")

    def sync_once(self) -> int:
        """Run one sync cycle. Return the number of MediaEvents delivered."""
        return self._sync() or 0

    def _sync(self) -> int | None:
        """Like ``sync_once`` but return None when ``/sync`` itself failed."""
        next_batch, encrypted_notified = self._load_state()
        initial_sync = next_batch is None
        query: dict[str, str] = {"timeout": str(self._config.poll_timeout_ms)}
        if next_batch:
            query["since"] = next_batch
        else:
            query["filter"] = json.dumps({"room": {"timeline": {"limit": 0}}}, separators=(",", ":"))
        # The HTTP timeout must outlast the server-side long-poll.
        timeout_sec = self._config.poll_timeout_ms / 1000.0 + 30.0
        resp = self._transport.request("GET", "/_matrix/client/v3/sync", query=query, timeout_sec=timeout_sec)
        if resp.status != 200 or resp.json is None:
            _log.warning("matrix subscriber: sync failed status=%s", resp.status)
            return None
        body = resp.json
        new_batch = body.get("next_batch")
        if not isinstance(new_batch, str):
            _log.warning("matrix subscriber: sync response without next_batch")
            return None
        delivered = 0
        rooms = body.get("rooms")
        if isinstance(rooms, dict):
            self._process_invites(rooms)
            joined = rooms.get("join")
            if isinstance(joined, dict) and not initial_sync:
                files_fn = self._download_file
                for room_id, block in joined.items():
                    if not isinstance(block, dict):
                        continue
                    # Encryption cannot be turned off again, so a notified room stays skipped.
                    if room_id in encrypted_notified or self._room_encrypted(block):
                        self._notify_encrypted(room_id, encrypted_notified)
                        continue
                    timeline = block.get("timeline")
                    if not isinstance(timeline, dict):
                        continue
                    events = timeline.get("events")
                    if not isinstance(events, list):
                        continue
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        event_id = ev.get("event_id")
                        if isinstance(event_id, str):
                            self._client.remember(event_id, room_id)
                        event = to_media_event(
                            ev,
                            room_id=room_id,
                            config=self._config,
                            files=files_fn,
                        )
                        if event is None:
                            continue
                        try:
                            self._handler.handle_event(event)
                        except Exception:
                            _log.exception(
                                "matrix subscriber: handle_event failed room=%s event=%s",
                                room_id,
                                ev.get("event_id"),
                            )
                        delivered += 1
        self._save_state(new_batch, encrypted_notified)
        return delivered

    def _interruptible_sleep(self, seconds: float) -> bool:
        return self._stop.wait(seconds)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ok = self._sync() is not None
            except Exception:
                _log.exception("matrix subscriber: sync failed")
                ok = False
            if ok:
                self._backoff_index = 0
            else:
                delay = _BACKOFF_SCHEDULE[min(self._backoff_index, len(_BACKOFF_SCHEDULE) - 1)]
                self._backoff_index = min(self._backoff_index + 1, len(_BACKOFF_SCHEDULE) - 1)
                if self._interruptible_sleep(delay):
                    return
                continue
            if self._interruptible_sleep(0.1):
                return
