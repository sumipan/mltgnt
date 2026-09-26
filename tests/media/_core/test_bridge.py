"""mltgnt.media._core.bridge (#4031): round trips over the Slack fake."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mltgnt.conversation import session_store, thread_queue
from mltgnt.interfaces.turn import HistoryMessage, TurnHandler, TurnInput, TurnResult
from mltgnt.media._core.bridge import MediaBridge
from mltgnt.media._core.hooks import HookRegistry
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.types import MediaEvent
from mltgnt.media._core.watchers import ExecDoneHandler
from mltgnt.media.slack.client import SlackClient
from mltgnt.media.slack.config import SlackMediaConfig
from tests.media.slack.fakes import FakeSlackApiError, FakeWebClient

UID = "11111111-1111-1111-1111-111111111111"
CID = "C1:100.0"
WORKING = "woman-raising-hand"
DONE = "ok_woman"
FAILED = "x"


@dataclass
class FakeFsEvent:
    src_path: str
    event_type: str = "created"
    is_directory: bool = False


class RecordingHandler:
    """Return ``results`` in order (the last one repeats); ``on_handle`` runs inside ``handle``."""

    def __init__(self, *results: TurnResult, on_handle=None, error: BaseException | None = None) -> None:
        self.results = list(results) or [TurnResult(kind="reply", text="pong")]
        self.turns: list[TurnInput] = []
        self.on_handle = on_handle
        self.error = error

    def handle(self, turn: TurnInput) -> TurnResult:
        self.turns.append(turn)
        if self.on_handle is not None:
            hook, self.on_handle = self.on_handle, None
            hook()
        if self.error is not None:
            raise self.error
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


@pytest.fixture(autouse=True)
def _conversation_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(thread_queue, "_config_provider", None)
    monkeypatch.setattr(session_store, "_sessions_dir_override", tmp_path / "sessions")


def _config(tmp_path: Path) -> SlackMediaConfig:
    return SlackMediaConfig(state_dir=tmp_path / "s", pending_dir=tmp_path / "p", events_dir=tmp_path / "e")


def _bridge(
    tmp_path: Path,
    handler: TurnHandler,
    web: FakeWebClient | None = None,
    hooks: HookRegistry | None = None,
) -> tuple[MediaBridge, FakeWebClient]:
    web = web or FakeWebClient()
    config = _config(tmp_path)
    client = SlackClient(web, config, default_channel="C1")
    return MediaBridge(client, handler, config, hooks or HookRegistry()), web


def _event(text: str = "ping", ts: str = "100.0", cid: str = CID) -> MediaEvent:
    return MediaEvent(space_id="C1", conversation_id=cid, message_id=ts, author="U1", text=text)


def _reactions(web: FakeWebClient, ts: str) -> list[str]:
    return [c["name"] for c in web.calls_of("reactions_add") if c["timestamp"] == ts]


def _roles(cid: str = CID) -> list[tuple[str, str]]:
    return [(t["role"], t["content"]) for t in session_store.load_turns(cid)]


def test_reply_round_trip(tmp_path: Path) -> None:
    handler = RecordingHandler(TurnResult(kind="reply", text="pong"))
    bridge, web = _bridge(tmp_path, handler)

    result = bridge.handle_event(_event())

    assert result == TurnResult(kind="reply", text="pong")
    assert web.calls_of("chat_postMessage") == [{"channel": "C1", "text": "pong", "thread_ts": "100.0"}]
    assert _roles() == [("user", "ping"), ("assistant", "pong")]
    assert _reactions(web, "100.0") == [WORKING, DONE]
    assert handler.turns[0].conversation_id == CID
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_history_carries_earlier_turns(tmp_path: Path) -> None:
    handler = RecordingHandler(TurnResult(kind="reply", text="pong"))
    bridge, _web = _bridge(tmp_path, handler)
    bridge.handle_event(_event("one", "100.0"))
    bridge.handle_event(_event("two", "101.0"))
    assert handler.turns[0].history == ()
    assert handler.turns[1].history == (
        HistoryMessage(role="user", text="one"),
        HistoryMessage(role="assistant", text="pong"),
    )


def test_delegation_then_delivery_via_watcher(tmp_path: Path) -> None:
    handler = RecordingHandler(TurnResult(kind="task", task_ref=UID))
    delivered: list[tuple[str, str]] = []
    hooks = HookRegistry()
    hooks.on_result(lambda uid, body: delivered.append((uid, body)))
    bridge, web = _bridge(tmp_path, handler, hooks=hooks)

    result = bridge.handle_event(_event())

    assert result == TurnResult(kind="task", task_ref=UID)
    pending = PendingStore(tmp_path / "p").load(UID)
    assert pending is not None
    assert pending["state"] == "running"
    assert (pending["space"], pending["thread"], pending["conversation_id"]) == ("C1", "100.0", CID)
    assert _reactions(web, "100.0")[-1] == WORKING
    assert web.calls_of("chat_postMessage") == []
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "running"

    ExecDoneHandler(lambda uid: bridge.deliver_result(uid, "job done")).dispatch(FakeFsEvent(f"/x/{UID}"))

    assert web.calls_of("chat_postMessage") == [{"channel": "C1", "text": "job done", "thread_ts": "100.0"}]
    assert delivered == [(UID, "job done")]
    assert _reactions(web, "100.0")[-1] == DONE
    assert _roles() == [("user", "ping"), ("assistant", "job done")]
    assert PendingStore(tmp_path / "p").load(UID) is None
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_deliver_result_twice_posts_once(tmp_path: Path) -> None:
    bridge, web = _bridge(tmp_path, RecordingHandler(TurnResult(kind="task", task_ref=UID)))
    bridge.handle_event(_event())
    assert bridge.deliver_result(UID, "body") == "1.000100"
    assert bridge.deliver_result(UID, "body") is None
    assert len(web.calls_of("chat_postMessage")) == 1


def test_deliver_result_keeps_pending_when_post_fails(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"chat_postMessage": FakeSlackApiError("channel_not_found")})
    delivered: list[str] = []
    hooks = HookRegistry()
    hooks.on_result(lambda uid, body: delivered.append(uid))
    bridge, _web = _bridge(tmp_path, RecordingHandler(TurnResult(kind="task", task_ref=UID)), web, hooks)
    bridge.handle_event(_event())
    assert bridge.deliver_result(UID, "body") is None
    assert PendingStore(tmp_path / "p").load(UID) is not None
    assert delivered == []


def test_task_without_ref_fails(tmp_path: Path) -> None:
    bridge, web = _bridge(tmp_path, RecordingHandler(TurnResult(kind="task")))
    assert bridge.handle_event(_event()) == TurnResult(kind="task")
    assert _reactions(web, "100.0")[-1] == FAILED
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_on_inbound_true_skips_handler(tmp_path: Path) -> None:
    handler = RecordingHandler()
    hooks = HookRegistry()
    hooks.on_inbound(lambda event: True)
    bridge, web = _bridge(tmp_path, handler, hooks=hooks)
    assert bridge.handle_event(_event()) is None
    assert handler.turns == []
    assert web.calls == []
    assert _roles() == []


def test_after_post_exception_does_not_stop_reply(tmp_path: Path) -> None:
    seen: list[str | None] = []
    hooks = HookRegistry()

    def boom(result: TurnResult, message_id: str | None) -> None:
        raise RuntimeError("boom")

    hooks.after_post(boom)
    hooks.after_post(lambda result, message_id: seen.append(message_id))
    bridge, _web = _bridge(tmp_path, RecordingHandler(TurnResult(kind="reply", text="pong")), hooks=hooks)
    assert bridge.handle_event(_event()) == TurnResult(kind="reply", text="pong")
    assert seen == ["1.000100"]


def test_before_dispatch_replaces_turn(tmp_path: Path) -> None:
    handler = RecordingHandler()
    hooks = HookRegistry()
    hooks.before_dispatch(lambda t: TurnInput(conversation_id=t.conversation_id, text="rewritten", persona_id="p1"))
    bridge, _web = _bridge(tmp_path, handler, hooks=hooks)
    bridge.handle_event(_event())
    assert handler.turns[0].text == "rewritten"
    assert session_store.load_turns(CID)[-1]["persona"] == "p1"


def test_second_event_while_delegated_is_queued(tmp_path: Path) -> None:
    handler = RecordingHandler(TurnResult(kind="task", task_ref=UID))
    bridge, web = _bridge(tmp_path, handler)
    bridge.handle_event(_event("first", "100.0"))
    assert bridge.handle_event(_event("second", "101.0")) is None
    assert len(handler.turns) == 1
    assert _reactions(web, "101.0") == [DONE]  # RECEIVED maps to the same reaction


def test_queued_message_runs_after_the_reply(tmp_path: Path) -> None:
    bridge_ref: list[MediaBridge] = []
    queued: list[TurnResult | None] = []
    handler = RecordingHandler(
        TurnResult(kind="reply", text="r1"),
        TurnResult(kind="reply", text="r2"),
        on_handle=lambda: queued.append(bridge_ref[0].handle_event(_event("second", "101.0"))),
    )
    bridge, web = _bridge(tmp_path, handler)
    bridge_ref.append(bridge)

    assert bridge.handle_event(_event("first", "100.0")) == TurnResult(kind="reply", text="r1")

    assert queued == [None]
    assert len(handler.turns) == 2
    assert "second" in handler.turns[1].text
    assert [p["text"] for p in web.calls_of("chat_postMessage")] == ["r1", "r2"]
    assert _reactions(web, "101.0")[-1] == DONE
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_queued_message_runs_after_delivery(tmp_path: Path) -> None:
    handler = RecordingHandler(TurnResult(kind="task", task_ref=UID), TurnResult(kind="reply", text="r2"))
    bridge, web = _bridge(tmp_path, handler)
    bridge.handle_event(_event("first", "100.0"))
    bridge.handle_event(_event("second", "101.0"))
    bridge.deliver_result(UID, "job done")
    assert len(handler.turns) == 2
    assert [p["text"] for p in web.calls_of("chat_postMessage")] == ["job done", "r2"]
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_handler_exception_marks_failed_and_releases(tmp_path: Path) -> None:
    bridge, web = _bridge(tmp_path, RecordingHandler(error=RuntimeError("boom")))
    assert bridge.handle_event(_event()) is None
    assert _reactions(web, "100.0")[-1] == FAILED
    assert web.calls_of("chat_postMessage") == []
    assert thread_queue.read_state(thread_queue.storage_key(CID))["status"] == "idle"


def test_reply_post_failure_marks_failed(tmp_path: Path) -> None:
    web = FakeWebClient(errors={"chat_postMessage": FakeSlackApiError("channel_not_found")})
    seen: list[str | None] = []
    hooks = HookRegistry()
    hooks.after_post(lambda result, message_id: seen.append(message_id))
    bridge, _web = _bridge(tmp_path, RecordingHandler(), web, hooks)
    assert bridge.handle_event(_event()) == TurnResult(kind="reply", text="pong")
    assert _reactions(web, "100.0")[-1] == FAILED
    assert _roles() == [("user", "ping")]
    assert seen == [None]


def test_opaque_conversation_id_posts_without_thread(tmp_path: Path) -> None:
    bridge, web = _bridge(tmp_path, RecordingHandler())
    bridge.handle_event(_event(cid="room-1"))
    assert web.calls_of("chat_postMessage") == [{"channel": "C1", "text": "pong"}]
