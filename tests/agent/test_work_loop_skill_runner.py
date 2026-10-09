"""Tests for work-loop skill tooling."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from mltgnt.agent.work_loop import (
    GhdagSkillRunner,
    WorkLoopConfig,
    make_skill_tool,
    run_skill_contract,
    run_work_loop,
)
from mltgnt.interfaces.types import ChatInput, Message
from mltgnt.skill.models import SkillMeta, SkillRunResult


def test_run_skill_contract() -> None:
    text = run_skill_contract(["a", "b"])
    assert "run_skill" in text
    assert "a" in text and "b" in text
    assert run_skill_contract([]) == ""


def test_make_skill_tool_allowed_and_errors() -> None:
    calls: list[tuple[str, str, str | None]] = []

    def runner(
        skill_name: str,
        arguments: str,
        *,
        parent_correlation_id: str | None,
    ) -> tuple[bool, str]:
        calls.append((skill_name, arguments, parent_correlation_id))
        return True, "body"

    inner_calls: list[tuple[str, dict]] = []

    def inner(tool: str, args: dict) -> str:
        inner_calls.append((tool, args))
        return "inner"

    wrapped = make_skill_tool(
        inner,
        runner,
        allowed=frozenset({"alpha"}),
        parent_correlation_id="parent-1",
    )
    assert wrapped("run_skill", {"skill": "alpha", "arguments": "hi"}) == "body"
    assert calls == [("alpha", "hi", "parent-1")]

    err = wrapped("run_skill", {"skill": "beta"})
    assert err == "[ERROR] skill not allowed: beta"
    assert len(calls) == 1

    assert wrapped("run_skill", {}).startswith("[ERROR]")
    assert wrapped("other", {"k": 1}) == "inner"
    assert inner_calls == [("other", {"k": 1})]

    calls.clear()

    def fail_runner(
        skill_name: str,
        arguments: str,
        *,
        parent_correlation_id: str | None,
    ) -> tuple[bool, str]:
        return False, "boom"

    wrapped2 = make_skill_tool(
        inner,
        fail_runner,
        allowed=frozenset({"alpha"}),
        parent_correlation_id=None,
    )
    assert wrapped2("run_skill", {"skill": "alpha"}) == "[ERROR] skill alpha failed: boom"


def test_run_work_loop_with_run_skill_integration() -> None:
    responses = iter(
        [
            '{"items": [{"id": "1", "title": "t"}]}',
            '{"tool": "run_skill", "args": {"skill": "nope"}}',
            '{"tool": "finish", "args": {"message": "done", "artifacts": []}}',
        ]
    )

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(responses)

    def runner(
        skill_name: str,
        arguments: str,
        *,
        parent_correlation_id: str | None,
    ) -> tuple[bool, str]:
        return True, "unused"

    outcome = run_work_loop(
        "order",
        llm_call=llm_call,
        tools=lambda _t, _a: "never",
        cfg=WorkLoopConfig(tool_skills=("allowed",)),
        skill_runner=runner,
        parent_correlation_id="corr",
    )
    assert outcome.status == "IMPL_DONE"
    assert outcome.message == "done"


def test_ghdag_skill_runner_enqueue(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: demo\ndescription: d\n---\n\nbody",
        encoding="utf-8",
    )
    meta = SkillMeta(name="demo", description="d", argument_hint="", model=None, path=skill_file)
    persona = MagicMock()
    persona.name = "persona-a"
    persona.format_prompt.return_value = "system prompt text"

    captured: dict[str, Any] = {}

    def fake_enqueue(**kwargs: Any) -> tuple[bool, str]:
        captured.update(kwargs)
        return True, "result-body"

    run_result = SkillRunResult(
        chat_input=ChatInput(
            source="work_loop",
            session_key="s",
            messages=[
                Message(role="system", content="system prompt text"),
                Message(role="user", content="argv"),
            ],
            persona_name="persona-a",
            model="m1",
        ),
        expected_markers=(),
        skill_io="legacy",
        produces=None,
    )

    with patch(
        "mltgnt.agent.work_loop.skill_runner_mod.run",
        return_value=run_result,
    ):
        runner = GhdagSkillRunner(
            {"demo": meta},
            persona,
            engine="cursor",
            model=None,
            jobs_dir=tmp_path / "jobs",
            exec_done_dir=tmp_path / "jobs" / "done",
            timeout=30.0,
            permission_by_skill={"demo": "allow"},
            enqueue=fake_enqueue,
        )
        ok, body = runner("demo", "argv", parent_correlation_id="parent-corr")
    assert ok is True and body == "result-body"
    assert captured["prompt"] == "system prompt text"
    assert captured["parent_correlation_id"] == "parent-corr"
    assert captured["persona_name"] == "persona-a"
    assert captured["permission"] == "allow"
    assert captured["run_result"] is run_result

    runner2 = GhdagSkillRunner(
        {},
        persona,
        engine="cursor",
        model=None,
        jobs_dir=tmp_path / "jobs",
        exec_done_dir=tmp_path / "jobs" / "done",
        timeout=30.0,
        enqueue=fake_enqueue,
    )
    ok2, msg = runner2("missing", "", parent_correlation_id=None)
    assert ok2 is False and msg == "skill not found: missing"
