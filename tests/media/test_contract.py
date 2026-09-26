"""Media contract (#4032): one TurnHandler gives the same turn on Slack and WebChat.

Reply, delegation, progress and result are run on the Slack fake and on WebChat
(store under ``tmp_path``); the TurnResult and what each medium shows must match.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mltgnt.conversation import session_store, thread_queue
from mltgnt.interfaces.media import MediaClient
from mltgnt.interfaces.turn import TurnInput, TurnResult
from mltgnt.media._core.bridge import MediaBridge
from mltgnt.media._core.pending import PendingStore
from mltgnt.media._core.progress import ProgressState, finalize_progress
from mltgnt.media._core.types import MediaEvent
from mltgnt.media.slack.client import SlackClient
from mltgnt.media.slack.config import SlackMediaConfig
from mltgnt.media.slack.inbound import to_media_event as slack_event
from mltgnt.media.webchat.client import WebChatClient
from mltgnt.media.webchat.config import WebChatMediaConfig
from mltgnt.media.webchat.inbound import to_media_event as webchat_event
from tests.media.slack.fakes import FakeWebClient

UID = "22222222-2222-2222-2222-222222222222"
TOOL_EVENT = {
    "type": "assistant",
    "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "notes.txt"}}]},
}


class ScriptedHandler:
    """The shared TurnHandler: returns ``result`` for every turn and records the inputs."""

    def __init__(self, result: TurnResult) -> None:
        self.result = result
        self.turns: list[TurnInput] = []

    def handle(self, turn: TurnInput) -> TurnResult:
        self.turns.append(turn)
        return self.result


@dataclass
class Medium:
    name: str
    client: MediaClient
    pending_dir: Path
    receive: Callable[[str], MediaEvent]
    posts: Callable[[], list[str]]
    shown: Callable[[str], str]
    bridge: MediaBridge | None = field(default=None)


def _slack(tmp_path: Path, handler: ScriptedHandler) -> Medium:
    root = tmp_path / "slack"
    config = SlackMediaConfig(state_dir=root / "s", pending_dir=root / "p", events_dir=root / "e")
    web = FakeWebClient()
    client = SlackClient(web, config, default_channel="C1")
    counter = itertools.count(100)

    def receive(text: str) -> MediaEvent:
        return slack_event({"type": "message", "channel": "C1", "user": "U1", "ts": f"{next(counter)}.0", "text": text})

    def shown(message_id: str) -> str:
        updates = [c["text"] for c in web.calls_of("chat_update") if c["ts"] == message_id]
        posted = [c["text"] for c in web.calls_of("chat_postMessage")]
        return updates[-1] if updates else posted[int(message_id.split(".")[0]) - 1]

    medium = Medium(
        "slack",
        client,
        config.pending_dir,
        receive,
        lambda: [c["text"] for c in web.calls_of("chat_postMessage")],
        shown,
    )
    medium.bridge = MediaBridge(client, handler, config)
    return medium


def _webchat(tmp_path: Path, handler: ScriptedHandler) -> Medium:
    root = tmp_path / "webchat"
    config = WebChatMediaConfig(
        state_dir=root / "s", pending_dir=root / "p", events_dir=root / "e", store_dir=root / "w"
    )
    client = WebChatClient(config)
    store = client.store
    counter = itertools.count(1)

    def receive(text: str) -> MediaEvent:
        # what POST /messages does: build the event, then store the incoming message
        event = webchat_event({"text": text, "author": "U1"}, space_id=config.space_id, message_id=f"in{next(counter)}")
        store.append(message_id=event.message_id, author=event.author, text=event.text)
        return event

    def shown(message_id: str) -> str:
        row = store.latest(message_id)
        return str(row["text"]) if row else ""

    medium = Medium(
        "webchat",
        client,
        config.pending_dir,
        receive,
        lambda: [
            str(row["text"])
            for day in store.days()
            for _, row in store.read_from(day)[0]
            if row["kind"] == "message" and row["author"] == "assistant"
        ],
        shown,
    )
    medium.bridge = MediaBridge(client, handler, config)
    return medium


FACTORIES = (_slack, _webchat)


@pytest.fixture(autouse=True)
def _conversation_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(thread_queue, "THREAD_QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(thread_queue, "_config_provider", None)
    monkeypatch.setattr(session_store, "_sessions_dir_override", tmp_path / "sessions")


def _media(tmp_path: Path, result: TurnResult) -> tuple[ScriptedHandler, list[Medium]]:
    handler = ScriptedHandler(result)
    return handler, [factory(tmp_path, handler) for factory in FACTORIES]


def _bridge(medium: Medium) -> MediaBridge:
    assert medium.bridge is not None
    return medium.bridge


def test_reply_is_the_same_on_both_media(tmp_path: Path) -> None:
    handler, media = _media(tmp_path, TurnResult(kind="reply", text="pong"))
    results = [_bridge(m).handle_event(m.receive("ping")) for m in media]
    assert results == [TurnResult(kind="reply", text="pong")] * 2
    assert [t.text for t in handler.turns] == ["ping", "ping"]
    for medium in media:
        assert medium.posts() == ["pong"], medium.name


def test_delegation_is_the_same_on_both_media(tmp_path: Path) -> None:
    _handler, media = _media(tmp_path, TurnResult(kind="task", task_ref=UID))
    results = [_bridge(m).handle_event(m.receive("do it")) for m in media]
    assert results == [TurnResult(kind="task", task_ref=UID)] * 2
    for medium in media:
        assert medium.posts() == [], medium.name
        pending = PendingStore(medium.pending_dir).load(UID)
        assert pending is not None and pending["state"] == "running", medium.name


def test_progress_is_the_same_on_both_media(tmp_path: Path) -> None:
    _handler, media = _media(tmp_path, TurnResult(kind="task", task_ref=UID))
    shown: list[list[str]] = []
    for medium in media:
        event = medium.receive("do it")
        assert _bridge(medium).handle_event(event) == TurnResult(kind="task", task_ref=UID)
        message_id = medium.client.post("...", event.space_id, event.conversation_id.partition(":")[2])
        assert message_id is not None
        state = ProgressState(min_interval_sec=0.0)
        state.process_event(TOOL_EVENT)
        assert state.maybe_update(medium.client, message_id, force=True) is True
        after_update = medium.shown(message_id)
        assert finalize_progress(medium.client, message_id, "0") is True
        shown.append([after_update, medium.shown(message_id)])
        assert medium.posts() == ["..."], medium.name
    assert shown[0] == shown[1]
    assert shown[0][0] == "Read notes.txt"


def test_result_is_the_same_on_both_media(tmp_path: Path) -> None:
    _handler, media = _media(tmp_path, TurnResult(kind="task", task_ref=UID))
    for medium in media:
        bridge = _bridge(medium)
        assert bridge.handle_event(medium.receive("do it")) == TurnResult(kind="task", task_ref=UID)
        assert bridge.deliver_result(UID, "job done") is not None
        assert bridge.deliver_result(UID, "job done") is None
        assert medium.posts() == ["job done"], medium.name
        assert PendingStore(medium.pending_dir).load(UID) is None, medium.name
