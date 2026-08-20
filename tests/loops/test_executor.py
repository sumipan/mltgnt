"""tests/loops/test_executor.py — GhdagSubtaskExecutor テスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from mltgnt.interfaces.loops import StepPoll, StepSubmission
from mltgnt.loops.executor import GhdagSubtaskExecutor


def test_executor_delegates_to_bridge(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    executor = GhdagSubtaskExecutor(
        jobs_dir=jobs,
        exec_done_dir=jobs / "done",
        engine="claude",
        model="",
    )
    sub = StepSubmission(
        uuid="u1",
        result_filename="r1.md",
        submitted_at="2026-08-20T12:00:00+09:00",
        reused=False,
    )
    poll = StepPoll(status="success", content="done")

    with (
        patch("mltgnt.loops.executor.enqueue_step", return_value=sub) as mock_enqueue,
        patch("mltgnt.loops.executor.poll_step", return_value=poll) as mock_poll,
    ):
        assert executor.submit(prompt="p", idempotency_key="k") is sub
        mock_enqueue.assert_called_once()
        assert executor.poll(uuid="u1", result_filename="r1.md") is poll
        mock_poll.assert_called_once()
