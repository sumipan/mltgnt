"""Unit tests for the BaseRunner ABC."""
from __future__ import annotations

import pytest

from mltgnt.scheduler.base_runner import BaseRunner
from mltgnt.scheduler.runner import PersonaScheduler


def test_cannot_instantiate_abstract() -> None:
    with pytest.raises(TypeError):
        BaseRunner()  # type: ignore[abstract]


def test_concrete_subclass_requires_tick() -> None:
    class ConcreteRunner(BaseRunner):
        def tick(self, now=None):
            return "ok"

    assert ConcreteRunner().tick() == "ok"


def test_persona_scheduler_is_base_runner() -> None:
    assert issubclass(PersonaScheduler, BaseRunner)


def test_incomplete_subclass_cannot_instantiate() -> None:
    class IncompleteRunner(BaseRunner):
        pass

    with pytest.raises(TypeError):
        IncompleteRunner()  # type: ignore[abstract]
