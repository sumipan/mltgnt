"""BaseRunner ABC の単体テスト。"""
from __future__ import annotations

import pytest

from mltgnt.execution import BaseRunner
from mltgnt.ooda.runner import OODARunner
from mltgnt.scheduler.runner import PersonaScheduler


def test_base_runner_is_abstract():
    with pytest.raises(TypeError):
        BaseRunner()  # type: ignore[abstract]


def test_concrete_subclass_is_instantiable():
    class ConcreteRunner(BaseRunner):
        def tick(self, now=None):
            return "ticked"

    runner = ConcreteRunner()
    assert runner.tick() == "ticked"


def test_persona_scheduler_is_base_runner():
    assert issubclass(PersonaScheduler, BaseRunner)


def test_ooda_runner_is_base_runner():
    assert issubclass(OODARunner, BaseRunner)


def test_persona_scheduler_instance_is_base_runner(tmp_path):
    sched = PersonaScheduler(slack=None, state_dir=tmp_path / "state")
    assert isinstance(sched, BaseRunner)


def test_subclass_without_tick_raises():
    class IncompleteRunner(BaseRunner):
        pass

    with pytest.raises(TypeError):
        IncompleteRunner()  # type: ignore[abstract]
