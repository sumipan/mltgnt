"""FastAPI app of the WebChat medium. FastAPI / uvicorn are imported lazily.

Endpoints: ``GET /`` (UI), ``GET /messages?day=YYYY-MM-DD``, ``POST /messages``,
``GET /threads/{thread_id}`` and ``GET /stream`` (SSE of ``message`` / ``update`` /
``status`` rows; the UI replaces the element with the same message_id).
SSE event ids are ``<day>:<byte offset>`` so a reconnect resumes via ``Last-Event-ID``.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import date
from typing import Any, Protocol

from mltgnt.media._core import id_map
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.media.webchat.inbound import to_media_event
from mltgnt.media.webchat.store import WebChatStore
from mltgnt.media.webchat.ui import INDEX_HTML

__all__ = ["EventHandler", "create_app", "parse_event_id", "serve"]

_log = logging.getLogger(__name__)

_EXTRA_HINT = "fastapi is not installed; install the extra: pip install 'mltgnt[webchat]'"


class EventHandler(Protocol):
    """What the app needs from ``MediaBridge``."""

    def handle_event(self, event: MediaEvent) -> Any: ...


def parse_event_id(value: str | None) -> tuple[date, int] | None:
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


def _sse(row: dict[str, Any], event_id: str) -> str:
    kind = row.get("kind") if row.get("kind") in ("message", "update", "status") else "message"
    return f"id: {event_id}\nevent: {kind}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"


def _run_event(bridge: EventHandler, event: MediaEvent) -> None:
    try:
        bridge.handle_event(event)
    except Exception:
        _log.exception("[webchat] handle_event failed message_id=%s", event.message_id)


def create_app(
    config: WebChatMediaConfig,
    bridge: EventHandler,
    *,
    store: WebChatStore | None = None,
    poll_interval_sec: float = 0.5,
    stream_timeout_sec: float | None = None,
) -> Any:
    """Build the FastAPI app. ``POST /messages`` stores the message, then runs the turn in the background.

    ``stream_timeout_sec`` closes ``GET /stream`` after that many seconds (the browser
    reconnects with ``Last-Event-ID``); None keeps it open until the client leaves.
    """
    try:
        from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
        from fastapi.responses import HTMLResponse, StreamingResponse
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    store = store if store is not None else WebChatStore(config.store_dir)
    app = FastAPI(title="mltgnt webchat")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(INDEX_HTML)

    @app.get("/messages")
    def list_messages(day: date | None = None) -> list[dict[str, Any]]:
        return store.read(day or store.today())

    @app.post("/messages", status_code=202)
    async def post_message(request: Request, background: BackgroundTasks) -> dict[str, Any]:
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="body must be JSON") from exc
        message_id = uuid.uuid4().hex
        try:
            event = to_media_event(body, space_id=config.space_id, message_id=message_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        thread = id_map.resolve(event.conversation_id)[1]
        thread_ts = None if thread == message_id else thread
        try:
            store.append(message_id=message_id, author=event.author, text=event.text, thread_ts=thread_ts)
        except OSError as exc:
            _log.warning("[webchat] store append failed: %s", exc)
            raise HTTPException(status_code=503, detail="store unavailable") from exc
        background.add_task(_run_event, bridge, event)
        return {"message_id": message_id, "thread_ts": thread_ts, "conversation_id": event.conversation_id}

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str) -> list[dict[str, Any]]:
        return store.thread(thread_id)

    @app.get("/stream")
    def stream(request: Request) -> Any:
        resume = parse_event_id(request.headers.get("last-event-id"))
        if resume is None:
            today = store.today()
            path = store.path_for(today)
            resume = (today, path.stat().st_size if path.is_file() else 0)
        return StreamingResponse(
            _tail(request, store, resume, poll_interval_sec, stream_timeout_sec),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


async def _tail(
    request: Any,
    store: WebChatStore,
    start: tuple[date, int],
    poll_interval_sec: float,
    timeout_sec: float | None,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> AsyncIterator[str]:
    """Yield SSE frames for rows appended after ``start``, following the day file across midnight."""
    day, offset = start
    deadline = None if timeout_sec is None else monotonic() + timeout_sec
    yield "retry: 1000\n\n"
    while True:
        # the day is taken before reading, so the old file is drained before switching
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


def serve(
    app: Any,
    config: WebChatMediaConfig,
    *,
    runner: Callable[..., Any] | None = None,
) -> None:
    """Run ``app`` on ``config.host:config.port`` (blocking). ``runner`` replaces ``uvicorn.run``."""
    if runner is None:
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError("uvicorn is not installed; install the extra: pip install 'mltgnt[webchat]'") from exc
        runner = uvicorn.run
    runner(app, host=config.host, port=config.port)
