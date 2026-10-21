"""FastAPI hub server: per-room JSONL stores and bearer auth."""

import asyncio
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator, Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from mltgnt import __version__
from mltgnt.exceptions import ConfigError
from mltgnt.interfaces.hub import HubAddress, HubMessage, _validate_identifier
from mltgnt.interfaces.media import Status
from mltgnt.interfaces.turn import Attachment
from mltgnt.media._core.jsonl_store import JsonlMessageStore, MESSAGE_KINDS
from mltgnt.media.hub.guard import GuardConfig, check

__all__ = [
    "HubServerConfig",
    "create_app",
    "load_server_config",
    "parse_cursor",
    "purge_expired_daily_files",
    "serve",
]

_log = logging.getLogger(__name__)
_EXTRA_HINT = "fastapi is not installed; install the extra: pip install 'mltgnt[hub]'"
_SSE_KINDS = frozenset(MESSAGE_KINDS)
_POST_BODY_KEYS = frozenset(
    {
        "id",
        "room",
        "thread",
        "sender",
        "to",
        "text",
        "attachments",
        "hop",
        "origin_id",
        "kind",
        "status",
    }
)
_PATCH_BODY_KEYS = frozenset({"kind", "text", "status"})


def parse_cursor(value: str | None) -> tuple[date, int] | None:
    """``<YYYY-MM-DD>:<offset>`` -> ``(day, offset)``; None when missing or malformed."""
    if not value:
        return None
    day_text, sep, offset_text = value.partition(":")
    if not sep:
        return None
    try:
        day = date.fromisoformat(day_text)
        offset = int(offset_text)
    except ValueError:
        return None
    if offset < 0:
        return None
    return day, offset


@dataclass(frozen=True)
class HubServerConfig:
    store_dir: Path
    host: str = "127.0.0.1"
    port: int = 8790
    rooms: tuple[str, ...] = ()
    token_env: Mapping[str, str] = field(default_factory=dict)
    guard: GuardConfig = field(default_factory=GuardConfig)
    ttl_days: int = 7


def _validate_port(port: int) -> None:
    if port < 1 or port > 65535:
        raise ConfigError(f"invalid port: {port}")


def _validate_rooms(rooms: list[str]) -> tuple[str, ...]:
    if not rooms:
        raise ConfigError("rooms must not be empty")
    seen: set[str] = set()
    out: list[str] = []
    for room in rooms:
        if not isinstance(room, str):
            raise ConfigError(f"invalid room: {room!r}")
        try:
            _validate_identifier(room, field="room")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if room in seen:
            raise ConfigError(f"duplicate room: {room!r}")
        seen.add(room)
        out.append(room)
    return tuple(out)


def _guard_from_mapping(data: Any) -> GuardConfig:
    if data is None:
        return GuardConfig()
    if not isinstance(data, dict):
        raise ConfigError("guard must be a mapping")
    try:
        return GuardConfig(
            hop_limit=int(data.get("hop_limit", 3)),
            max_agent_run=int(data.get("max_agent_run", 4)),
            min_interval_sec=float(data.get("min_interval_sec", 5.0)),
            room_daily_limit=int(data.get("room_daily_limit", 200)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid guard: {exc}") from exc


def load_server_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> HubServerConfig:
    """Load YAML config and verify token env vars are set (values are not stored in config)."""
    env = environ if environ is not None else os.environ
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    store_dir = raw.get("store_dir")
    if not isinstance(store_dir, str) or not store_dir:
        raise ConfigError("store_dir is required")
    host = raw.get("host", "127.0.0.1")
    if not isinstance(host, str):
        raise ConfigError("host must be a string")
    port = raw.get("port", 8790)
    if not isinstance(port, int):
        raise ConfigError("port must be an integer")
    _validate_port(port)
    rooms_raw = raw.get("rooms")
    if not isinstance(rooms_raw, list):
        raise ConfigError("rooms must be a list")
    rooms = _validate_rooms(rooms_raw)
    ttl_days = raw.get("ttl_days", 7)
    if not isinstance(ttl_days, int) or ttl_days < 1:
        raise ConfigError("ttl_days must be an integer >= 1")
    token_env_raw = raw.get("token_env")
    if not isinstance(token_env_raw, dict) or not token_env_raw:
        raise ConfigError("token_env must be a non-empty mapping")
    token_env: dict[str, str] = {}
    for host_id, var_name in token_env_raw.items():
        if not isinstance(host_id, str) or not isinstance(var_name, str):
            raise ConfigError("token_env keys and values must be strings")
        try:
            _validate_identifier(host_id, field="host")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if var_name not in env or not env[var_name]:
            raise ConfigError(f"missing environment variable: {var_name}")
        token_env[host_id] = var_name
    guard = _guard_from_mapping(raw.get("guard"))
    return HubServerConfig(
        store_dir=Path(store_dir),
        host=host,
        port=port,
        rooms=rooms,
        token_env=token_env,
        guard=guard,
        ttl_days=ttl_days,
    )


def purge_expired_daily_files(room_dir: Path, *, ttl_days: int, today: date) -> None:
    """Remove ``YYYY-MM-DD.jsonl`` files older than ``ttl_days``; leave other names untouched."""
    if not room_dir.is_dir():
        return
    cutoff = today.fromordinal(today.toordinal() - ttl_days)
    for path in room_dir.iterdir():
        if not path.is_file() or path.suffix != ".jsonl":
            continue
        try:
            day = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if day < cutoff:
            path.unlink(missing_ok=True)


def _resolve_tokens(config: HubServerConfig, environ: Mapping[str, str]) -> dict[str, str]:
    """Map bearer token -> host_id."""
    out: dict[str, str] = {}
    for host_id, var_name in config.token_env.items():
        token = environ[var_name]
        out[token] = host_id
    return out


def _sse(row: dict[str, Any], event_id: str) -> str:
    kind = row.get("kind") if row.get("kind") in _SSE_KINDS else "message"
    return f"id: {event_id}\nevent: {kind}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"


def _attachments_from_body(items: Any) -> tuple[Attachment, ...]:
    if items is None:
        return ()
    if not isinstance(items, list):
        raise ValueError("attachments must be a list")
    out: list[Attachment] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid attachments")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("invalid attachments")
        out.append(
            Attachment(
                name=name,
                content_type=item.get("content_type"),
                uri=item.get("uri"),
            )
        )
    return tuple(out)


def _address_from_body(data: Any, *, field: str) -> HubAddress | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError(f"invalid {field}")
    host = data.get("host")
    if not isinstance(host, str):
        raise ValueError(f"invalid {field}")
    persona = data.get("persona")
    if persona is not None and not isinstance(persona, str):
        raise ValueError(f"invalid {field}")
    return HubAddress(host, persona)


def _hub_message_from_post(body: dict[str, Any], *, room: str) -> HubMessage:
    extra = set(body.keys()) - _POST_BODY_KEYS
    if extra:
        raise ValueError(f"unexpected fields: {sorted(extra)!r}")
    message_id = body.get("id")
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("id is required")
    if body.get("room") != room:
        raise ValueError("room mismatch")
    sender = _address_from_body(body.get("sender"), field="sender")
    if sender is None:
        raise ValueError("sender is required")
    kind = body.get("kind", "message")
    if kind not in ("message", "status", "update"):
        raise ValueError("invalid kind")
    text = body.get("text")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if kind == "message" and not text:
        raise ValueError("text must not be empty")
    hop = body.get("hop", 0)
    if not isinstance(hop, int) or hop < 0:
        raise ValueError("invalid hop")
    thread = body.get("thread")
    if thread is not None and not isinstance(thread, str):
        raise ValueError("invalid thread")
    status = body.get("status")
    if status is not None and not isinstance(status, str):
        raise ValueError("invalid status")
    origin_id = body.get("origin_id")
    if origin_id is not None and not isinstance(origin_id, str):
        raise ValueError("invalid origin_id")
    return HubMessage(
        id=message_id,
        ts=None,
        room=room,
        thread=thread,
        sender=sender,
        to=_address_from_body(body.get("to"), field="to"),
        text=text,
        attachments=_attachments_from_body(body.get("attachments")),
        hop=hop,
        origin_id=origin_id,
        kind=kind,
        status=status,
    )


def _normalized_row_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    msg = HubMessage.from_row(row)
    return {
        "room": msg.room,
        "thread": msg.thread,
        "text": msg.text,
        "to": msg.to.host if msg.to else None,
        "to_persona": msg.to.persona if msg.to else None,
        "hop": msg.hop,
        "origin_id": msg.origin_id,
        "kind": msg.kind,
        "status": msg.status,
        "attachments": tuple((a.name, a.content_type, a.uri) for a in msg.attachments),
    }


def _message_signature(message: HubMessage) -> dict[str, Any]:
    return {
        "room": message.room,
        "thread": message.thread,
        "text": message.text,
        "to": message.to.host if message.to else None,
        "to_persona": message.to.persona if message.to else None,
        "hop": message.hop,
        "origin_id": message.origin_id,
        "kind": message.kind,
        "status": message.status,
        "attachments": tuple((a.name, a.content_type, a.uri) for a in message.attachments),
    }


def _thread_display_rows(store: JsonlMessageStore, thread_id: str | None) -> list[dict[str, Any]]:
    if thread_id is None:
        return []
    return [row for row in store.thread(thread_id) if row.get("kind") in MESSAGE_KINDS]


def _count_today_messages(store: JsonlMessageStore) -> int:
    return sum(1 for row in store._rows(store.today()) if row.get("kind") == "message")


def _iter_from_cursor(
    stores: Mapping[str, JsonlMessageStore],
    room: str,
    start: tuple[date, int],
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], str | None]:
    store = stores[room]
    day, offset = start
    out: list[dict[str, Any]] = []
    last_cursor: str | None = None
    while len(out) < limit:
        rows, offset = store.read_from(day, offset)
        for end, row in rows:
            out.append(row)
            last_cursor = f"{day.isoformat()}:{end}"
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
        today = store.today()
        if today > day:
            day, offset = today, 0
            continue
        break
    return out, last_cursor


async def _tail(
    request: Any,
    store: JsonlMessageStore,
    start: tuple[date, int],
    poll_interval_sec: float,
    timeout_sec: float | None,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    day, offset = start
    deadline = None if timeout_sec is None else monotonic() + timeout_sec
    yield "retry: 1000\n\n"
    while True:
        today = store.today()
        rows, offset = store.read_from(day, offset)
        for end, row in rows:
            yield _sse(row, f"{day.isoformat()}:{end}")
        if today > day:
            day, offset = today, 0
            continue
        if deadline is not None and monotonic() >= deadline:
            return
        if await request.is_disconnected():
            return
        await asyncio.sleep(poll_interval_sec)


def create_app(
    config: HubServerConfig,
    *,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], datetime] | None = None,
    poll_interval_sec: float = 0.5,
    stream_timeout_sec: float | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    env = environ if environ is not None else os.environ
    token_to_host = _resolve_tokens(config, env)
    stores: MutableMapping[str, JsonlMessageStore] = {}
    for room in config.rooms:
        room_dir = config.store_dir / room
        room_dir.mkdir(parents=True, exist_ok=True)
        store = JsonlMessageStore(room_dir, clock=clock) if clock is not None else JsonlMessageStore(room_dir)
        purge_expired_daily_files(room_dir, ttl_days=config.ttl_days, today=store.today())
        stores[room] = store

    app = FastAPI(title="mltgnt hub")

    def _auth_host(request: Request) -> str:
        header = request.headers.get("authorization")
        if not header or not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = header[7:].strip()
        for candidate, host_id in token_to_host.items():
            if secrets.compare_digest(token, candidate):
                return host_id
        raise HTTPException(status_code=401, detail="invalid bearer token")

    def _room_or_404(room: str) -> str:
        if room not in stores:
            raise HTTPException(status_code=404, detail="unknown room")
        return room

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"rooms": list(config.rooms), "version": __version__}

    @app.post("/rooms/{room}/messages")
    async def post_message(request: Request, room: str) -> Any:
        auth_host = _auth_host(request)
        room = _room_or_404(room)
        store = stores[room]
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        try:
            message = _hub_message_from_post(body, room=room)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if message.sender.host != auth_host:
            raise HTTPException(status_code=403, detail="sender.host does not match token")
        existing = store.latest(message.id)
        if existing is not None:
            if _normalized_row_signature(existing) == _message_signature(message):
                return JSONResponse(status_code=200, content={"id": message.id, "ts": existing.get("ts")})
            raise HTTPException(status_code=409, detail="id_conflict")
        if message.thread is not None:
            guard_rows = _thread_display_rows(store, message.thread)
        else:
            guard_rows = [r for r in store._rows(store.today()) if r.get("kind") == "message"]
        now = store._clock()
        verdict = check(
            message,
            guard_rows,
            _count_today_messages(store),
            now,
            config=config.guard,
        )
        if not verdict.ok:
            raise HTTPException(status_code=409, detail={"reason": verdict.reason})
        row_extra = {
            k: v
            for k, v in message.to_row().items()
            if k
            not in (
                "ts",
                "message_id",
                "author",
                "text",
                "thread_ts",
                "kind",
                "status",
                "files",
            )
        }
        try:
            row = store.append(
                message_id=message.id,
                author=message.to_row()["author"],
                text=message.text,
                thread_ts=message.thread,
                kind=message.kind,
                status=message.status,
                files=message.to_row().get("files"),
                extra=row_extra,
            )
        except OSError as exc:
            _log.warning("[hub] store append failed: %s", exc)
            raise HTTPException(status_code=503, detail="store unavailable") from exc
        purge_expired_daily_files(store._dir, ttl_days=config.ttl_days, today=store.today())
        return JSONResponse(status_code=201, content={"id": message.id, "ts": row["ts"]})

    @app.patch("/rooms/{room}/messages/{message_id}")
    async def patch_message(request: Request, room: str, message_id: str) -> Any:
        auth_host = _auth_host(request)
        room = _room_or_404(room)
        store = stores[room]
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="body must be JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="body must be an object")
        extra = set(body.keys()) - _PATCH_BODY_KEYS
        if extra:
            raise HTTPException(status_code=422, detail=f"unexpected fields: {sorted(extra)!r}")
        kind = body.get("kind")
        if kind not in ("update", "status"):
            raise HTTPException(status_code=422, detail="kind must be update or status")
        current = store.latest(message_id)
        if current is None:
            raise HTTPException(status_code=404, detail="unknown message id")
        author = current.get("author")
        if not isinstance(author, str) or not author.startswith(f"{auth_host}/"):
            raise HTTPException(status_code=403, detail="forbidden")
        if kind == "update":
            text = body.get("text")
            if not isinstance(text, str) or not text:
                raise HTTPException(status_code=422, detail="text is required for update")
            changes: dict[str, Any] = {"text": text}
        else:
            status_val = body.get("status")
            if status_val is None:
                raise HTTPException(status_code=422, detail="status is required")
            if isinstance(status_val, Status):
                status_str = status_val.value
            elif isinstance(status_val, str):
                status_str = status_val
            else:
                raise HTTPException(status_code=422, detail="invalid status")
            changes = {"status": status_str}
        try:
            row = store.revise(message_id, kind, **changes)
        except OSError as exc:
            _log.warning("[hub] store revise failed: %s", exc)
            raise HTTPException(status_code=503, detail="store unavailable") from exc
        if row is None:
            raise HTTPException(status_code=404, detail="unknown message id")
        return {"id": message_id, "ts": row["ts"]}

    @app.get("/rooms/{room}/messages")
    def list_messages(
        request: Request,
        room: str,
        after: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        _auth_host(request)
        room = _room_or_404(room)
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
        store = stores[room]
        if after is None:
            today = store.today()
            start = (today, 0)
        else:
            parsed = parse_cursor(after)
            if parsed is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
            start = parsed
        messages, cursor = _iter_from_cursor(stores, room, start, limit=limit)
        return {"messages": messages, "cursor": cursor}

    @app.get("/rooms/{room}/guard-context")
    def guard_context(request: Request, room: str, thread: str | None = None) -> dict[str, Any]:
        _auth_host(request)
        room = _room_or_404(room)
        store = stores[room]
        rows = _thread_display_rows(store, thread)
        return {"thread_rows": rows, "room_today_count": _count_today_messages(store)}

    @app.get("/rooms/{room}/threads/{thread_id}")
    def get_thread(request: Request, room: str, thread_id: str) -> list[dict[str, Any]]:
        _auth_host(request)
        room = _room_or_404(room)
        store = stores[room]
        return _thread_display_rows(store, thread_id)

    @app.get("/rooms/{room}/stream")
    def stream(request: Request, room: str) -> Any:
        _auth_host(request)
        room = _room_or_404(room)
        store = stores[room]
        header = request.headers.get("last-event-id")
        if header:
            resume = parse_cursor(header)
            if resume is None:
                raise HTTPException(status_code=422, detail="invalid cursor")
        else:
            today = store.today()
            path = store.path_for(today)
            resume = (today, path.stat().st_size if path.is_file() else 0)
        return StreamingResponse(
            _tail(request, store, resume, poll_interval_sec, stream_timeout_sec),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def serve(
    config: HubServerConfig,
    *,
    environ: Mapping[str, str] | None = None,
    runner: Callable[..., Any] | None = None,
) -> None:
    """Run the hub app on ``config.host:config.port`` (blocking)."""
    app = create_app(config, environ=environ)
    if runner is None:
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError("uvicorn is not installed; install the extra: pip install 'mltgnt[hub]'") from exc
        runner = uvicorn.run
    runner(app, host=config.host, port=config.port)
