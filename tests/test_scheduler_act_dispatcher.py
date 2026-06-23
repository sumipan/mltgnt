"""PersonaScheduler ディスパッチ統一テスト。"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.scheduler.models import ScheduleJob
from mltgnt.scheduler.runner import PersonaScheduler


def _make_job(action: str, **kwargs: Any) -> ScheduleJob:
    return ScheduleJob(
        id=f"test-{action}",
        action=action,
        mode=kwargs.pop("mode", "scheduled"),
        notify=kwargs.pop("notify", "silent"),
        **kwargs,
    )


def test_skill_action_dispatched_via_actions_dict(tmp_path):
    """skill アクションが _actions 辞書経由でディスパッチされること。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    assert "skill" in sched._actions


def test_skill_action_registered_on_init(tmp_path):
    """__init__ 時点で skill が _actions に登録済みであること。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    skill_fn = sched._actions.get("skill")
    assert skill_fn is not None
    assert callable(skill_fn)


def test_execute_action_skill_calls_registered_handler(tmp_path):
    """execute_action('skill') が _actions['skill'] 経由で呼ばれること。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    mock_fn = MagicMock(return_value=(True, "skill_ok"))
    sched.register_action("skill", mock_fn)

    job = _make_job("skill")
    result = sched.execute_action(job)

    mock_fn.assert_called_once_with(job)
    assert result == (True, "skill_ok")


def test_execute_action_noop_returns_immediately(tmp_path):
    """noop は _actions 辞書を経由せず即座に (True, '') を返すこと。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    job = _make_job("noop")
    ok, msg = sched.execute_action(job)
    assert ok is True
    assert msg == ""


def test_execute_action_unknown_raises(tmp_path):
    """未登録アクションは ValueError を送出すること。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    job = _make_job("unknown_action_xyz")
    with pytest.raises(ValueError, match="未対応"):
        sched.execute_action(job)


def test_execute_action_custom_action_dispatched(tmp_path):
    """カスタムアクションも _actions 辞書経由でディスパッチされること。"""
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    called = []

    def my_action(job: ScheduleJob) -> tuple[bool, str]:
        called.append(job.id)
        return True, "custom_ok"

    sched.register_action("my_custom", my_action)
    job = _make_job("my_custom")
    ok, msg = sched.execute_action(job)
    assert ok is True
    assert msg == "custom_ok"
    assert called == [f"test-my_custom"]


def test_run_skill_action_not_called_directly_in_execute_action(tmp_path):
    """execute_action() が run_skill_action を直接呼ばず _actions 経由であること。"""
    import inspect
    import mltgnt.scheduler.runner as runner_mod

    src = inspect.getsource(PersonaScheduler.execute_action)
    assert "run_skill_action" not in src, (
        "execute_action() が run_skill_action() を直接呼び出しています。"
        "_actions 辞書経由に統一してください。"
    )
