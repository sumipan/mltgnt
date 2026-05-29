"""ExecAppenderDispatcher のユニットテスト。"""
from __future__ import annotations

import json

from mltgnt.ooda.exec_dispatcher import ExecAppenderDispatcher


class _FakeSlack:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def post_message(
        self,
        text: str,
        channel: str,
        thread_ts: str | None = None,
        blocks: list[dict] | None = None,
        reply_broadcast: bool = False,
    ) -> bool:
        self.messages.append((channel, text))
        return True


def test_skip_returns_success_without_side_effects(tmp_path):
    exec_path = tmp_path / "exec.jsonl"
    dispatcher = ExecAppenderDispatcher(exec_jsonl_path=exec_path)
    result = dispatcher.dispatch("skip", {})
    assert result.action == "skip"
    assert result.success is True
    assert not exec_path.exists()


def test_recover_task_appends_valid_json_line(tmp_path):
    exec_path = tmp_path / "exec.jsonl"
    dispatcher = ExecAppenderDispatcher(exec_jsonl_path=exec_path)
    result = dispatcher.dispatch(
        "recover_task",
        {
            "uuid": "task-uuid-1",
            "command": "claude -p < jobs/order.md",
            "result_path": "jobs/result.md",
            "idempotency_key": "ooda-recover:100",
        },
    )
    assert result.success is True
    lines = [ln for ln in exec_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["uuid"] == "task-uuid-1"
    assert record["idempotency_key"] == "ooda-recover:100"
    assert "command" in record


def test_escalate_to_slack_posts_message(tmp_path):
    slack = _FakeSlack()
    dispatcher = ExecAppenderDispatcher(
        exec_jsonl_path=tmp_path / "exec.jsonl",
        slack_client=slack,
        default_channel="C_TEST",
    )
    result = dispatcher.dispatch(
        "escalate_to_slack",
        {"text": "help needed", "channel": "C_TEST"},
    )
    assert result.success is True
    assert slack.messages == [("C_TEST", "help needed")]
