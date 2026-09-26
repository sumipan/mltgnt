"""mltgnt.media._core.progress (#4030)."""

from __future__ import annotations

import json
from pathlib import Path

from mltgnt.config.language import JA
from mltgnt.interfaces.media import Status
from mltgnt.media._core.config import MediaConfig
from mltgnt.media._core.progress import (
    ProgressState,
    finalize_progress,
    status_for_done_marker,
    summarize_tool_use_block,
    summarize_work_loop_step,
)


class FakeClient:
    def __init__(self, ok: bool = True) -> None:
        self.updates: list[tuple[str, str]] = []
        self.ok = ok

    def post(self, text: str, space: str, thread: str | None = None) -> str | None:
        return None

    def update(self, message_id: str, text: str) -> bool:
        self.updates.append((message_id, text))
        return self.ok

    def set_status(self, message_id: str, status: Status) -> bool:
        return True


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _text_event(text: str) -> dict:
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def test_burst_of_five_lines_updates_once_with_last_line(tmp_path: Path) -> None:
    config = MediaConfig(state_dir=tmp_path, pending_dir=tmp_path, events_dir=tmp_path, progress_min_interval_sec=3.0)
    clock = FakeClock()
    state = ProgressState.from_config(config, clock=clock)
    client = FakeClient()
    for i in range(5):
        state.process_event(_text_event(f"[progress] step {i}"))
        state.maybe_update(client, "m1")
        clock.now += 0.2
    assert client.updates == []
    clock.now = 100.0 + 3.0
    assert state.maybe_update(client, "m1") is True
    state.maybe_update(client, "m1")
    assert client.updates == [("m1", "step 4")]


def test_update_waits_interval_after_previous_update() -> None:
    clock = FakeClock()
    state = ProgressState(min_interval_sec=3.0, clock=clock)
    client = FakeClient()
    state.process_event(_text_event("[progress] a"))
    assert state.maybe_update(client, "m", force=True) is True
    state.process_event(_text_event("[progress] b"))
    clock.now += 1.0
    assert state.maybe_update(client, "m") is False
    clock.now += 2.0
    assert state.maybe_update(client, "m") is True
    assert [t for _, t in client.updates] == ["a", "b"]


def test_failed_update_is_retried() -> None:
    clock = FakeClock()
    state = ProgressState(min_interval_sec=0.0, clock=clock)
    client = FakeClient(ok=False)
    state.process_event(_text_event("[progress] a"))
    assert state.maybe_update(client, "m") is False
    client.ok = True
    assert state.maybe_update(client, "m") is True


def test_text_without_progress_marker_is_ignored() -> None:
    state = ProgressState(min_interval_sec=0.0)
    assert state.process_event(_text_event("just thinking out loud")) == []
    assert state.process_event({"type": "thinking"}) == []
    assert state.process_event({"type": "assistant", "message": "bad"}) == []
    assert state.render_text() == ""


def test_max_lines_keeps_latest_lines() -> None:
    state = ProgressState(min_interval_sec=0.0, max_lines=2)
    state.process_event(_text_event("[progress] a\n[progress]: b\n[progress] c"))
    assert state.render_text() == "b\nc"


def test_tool_use_and_work_loop_lines() -> None:
    state = ProgressState(min_interval_sec=0.0, max_lines=5)
    state.process_event(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "cd x && FOO=1 pytest -q"}},
                ]
            },
        }
    )
    state.process_event({"type": "work_loop_step", "tool": "fetch", "args": {"url": "u"}, "plan": [1, 3]})
    assert state.render_text() == "Read a.py\nBash pytest\nfetch u (plan 1/3)"


def test_summaries() -> None:
    assert summarize_tool_use_block({"name": "Bash", "input": {"description": "Run  tests"}}) == "Bash Run tests"
    assert summarize_tool_use_block({"name": "Bash", "input": {"command": "cd a"}}) == "Bash cd"
    assert summarize_tool_use_block({"name": "Bash", "input": {"command": "echo 'x"}}) == "Bash"
    assert summarize_tool_use_block({"name": "Bash", "input": {}}) == "Bash"
    assert summarize_tool_use_block({"name": "Grep", "input": None}) == "Grep"
    assert summarize_tool_use_block({"input": {}}) == ""
    assert summarize_work_loop_step({"tool": "t", "args": "bad", "plan": [True, 2]}) == "t"
    assert summarize_work_loop_step({}) == ""


def test_read_new_lines_tails_the_file(tmp_path: Path) -> None:
    path = tmp_path / "e.jsonl"
    state = ProgressState(min_interval_sec=0.0)
    assert state.read_new_lines(path) is False
    path.write_text(json.dumps(_text_event("[progress] one")) + "\nnot json\n[1]\n", encoding="utf-8")
    assert state.read_new_lines(path) is True
    assert state.read_new_lines(path) is False
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_text_event("[progress] two")) + "\n")
    assert state.read_new_lines(path) is True
    assert state.render_text() == "two"


def test_finalize_uses_status_labels() -> None:
    client = FakeClient()
    assert finalize_progress(client, None, "0") is False
    assert finalize_progress(client, "m", "0") is True
    finalize_progress(client, "m", "CANCELLED")
    finalize_progress(client, "m", "1")
    labels = JA.status_labels
    assert [t for _, t in client.updates] == [labels["done"], labels["cancelled"], labels["failed"]]
    assert status_for_done_marker("x") is Status.FAILED
