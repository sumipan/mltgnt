"""
tests/test_skill/test_runner.py — runner.run / run_agent のユニットテスト。

設計: Issue #124 §8 AC-4, AC-5 / Issue #907
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock


from mltgnt.chat.models import ChatInput
from mltgnt.skill.models import SkillFile, SkillMeta
from mltgnt.skill.runner import run, run_agent


def _make_skill(
    body: str,
    model: str | None = None,
    name: str = "review",
    agent: bool = False,
    tools: list[dict] | None = None,
) -> SkillFile:
    meta = SkillMeta(
        name=name,
        description="test",
        argument_hint="",
        model=model,
        path=Path("/fake/skills/review/SKILL.md"),
        agent=agent,
        tools=tools or [],
    )
    return SkillFile(meta=meta, body=body)


def _make_persona(name: str = "タチコマ") -> MagicMock:
    persona = MagicMock()
    persona.name = name
    persona.format_prompt = lambda instruction: f"[PERSONA:{name}]\n{instruction}"
    return persona


def _make_chat_input(model: str | None = "default-model") -> ChatInput:
    return ChatInput(
        source="test",
        session_key="session-1",
        messages=[{"role": "user", "content": "hello"}],
        model=model,
    )


class TestRunVariableSubstitution:
    def test_arguments_and_positional(self) -> None:
        """AC-4-1: $ARGUMENTS, $0, $1 置換"""
        skill = _make_skill("file=$0 mode=$1 all=$ARGUMENTS")
        persona = _make_persona()
        result = run(skill, persona, "日記/2026-04-17.md critique", _make_chat_input())
        sys_content = result.messages[0]["content"]
        assert "日記/2026-04-17.md critique" in sys_content  # $ARGUMENTS
        assert "file=日記/2026-04-17.md" in sys_content       # $0
        assert "mode=critique" in sys_content                  # $1

    def test_empty_arguments(self) -> None:
        """AC-4-2: 空引数 → $ARGUMENTS → "", $0 → ""」"""
        skill = _make_skill("args=[$ARGUMENTS] pos=[$0]")
        persona = _make_persona()
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.messages[0]["content"]
        assert "args=[]" in sys_content
        assert "pos=[]" in sys_content

    def test_persona_substitution(self) -> None:
        """AC-4-3: $PERSONA → ペルソナ名"""
        skill = _make_skill("persona=$PERSONA")
        persona = _make_persona("タチコマ")
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.messages[0]["content"]
        assert "タチコマ" in sys_content

    def test_skill_dir_substitution(self) -> None:
        """AC-4-4: $SKILL_DIR → SKILL.md の親ディレクトリ"""
        skill = _make_skill("dir=$SKILL_DIR")
        persona = _make_persona()
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.messages[0]["content"]
        assert "/fake/skills/review" in sys_content

    def test_out_of_range_positional(self) -> None:
        """AC-4-5: $3 だが引数が2つ → 空文字"""
        skill = _make_skill("$3")
        persona = _make_persona()
        result = run(skill, persona, "a b", _make_chat_input())
        sys_content = result.messages[0]["content"]
        assert "$3" not in sys_content  # 置換されている
        # $3 → "" なのでその部分は空
        assert "[PERSONA:" in sys_content


class TestRunPromptComposition:
    def test_system_prompt_contains_persona_and_skill(self) -> None:
        """AC-5-1: システムプロンプトにペルソナ指示とスキル本文が含まれる"""
        skill = _make_skill("スキル本文")
        persona = _make_persona("タチコマ")
        result = run(skill, persona, "", _make_chat_input())
        assert result.messages[0]["role"] == "system"
        sys_content = result.messages[0]["content"]
        assert "タチコマ" in sys_content
        assert "スキル本文" in sys_content

    def test_skill_model_overrides(self) -> None:
        """AC-5-2: skill.meta.model が指定されれば返却 model がスキル側の値"""
        skill = _make_skill("body", model="claude-opus-4-6")
        result = run(skill, _make_persona(), "", _make_chat_input(model="default-model"))
        assert result.model == "claude-opus-4-6"

    def test_null_model_inherits(self) -> None:
        """AC-5-3: skill.meta.model が null なら chat_input.model を引き継ぐ"""
        skill = _make_skill("body", model=None)
        result = run(skill, _make_persona(), "", _make_chat_input(model="default-model"))
        assert result.model == "default-model"

    def test_user_message_preserved(self) -> None:
        """元のユーザーメッセージが引き継がれる"""
        skill = _make_skill("body")
        chat_input = _make_chat_input()
        result = run(skill, _make_persona(), "", chat_input)
        user_msgs = [m for m in result.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hello"

    def test_original_system_message_replaced(self) -> None:
        """元のシステムメッセージは新しいシステムプロンプトに差し替えられる"""
        skill = _make_skill("new system")
        chat_input = ChatInput(
            source="test",
            session_key="s",
            messages=[
                {"role": "system", "content": "old system"},
                {"role": "user", "content": "hi"},
            ],
            model=None,
        )
        result = run(skill, _make_persona(), "", chat_input)
        system_msgs = [m for m in result.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "old system" not in system_msgs[0]["content"]
        assert "new system" in system_msgs[0]["content"]


def _make_llm(responses: list):
    calls = iter(responses)

    def llm_call(prompt: str, *, tool_result: str | None = None) -> str | None:
        return next(calls)

    return llm_call


def _make_executor(results: dict):
    def executor(tool_name: str, tool_args: dict) -> str:
        return results[tool_name]
    return executor


class TestRunAgent:
    """#907: run_agent — skill から AgentRunner ループへの統合。"""

    def test_agent_skill_runs_terminal_tool(self) -> None:
        """agent: true のスキルが AgentRunner ループで終端ツールまで到達する。"""
        skill = _make_skill(
            "あなたはエージェントです",
            agent=True,
            tools=[{"name": "slack_reply", "description": "Slackに返信"}],
        )
        persona = _make_persona()
        chat_input = ChatInput(
            source="test", session_key="s",
            messages=[{"role": "user", "content": "こんにちは"}],
            model=None,
        )
        llm = _make_llm(['{"tool": "slack_reply", "args": {"message": "hello"}}'])
        result = run_agent(
            skill, persona, "", chat_input,
            llm_call=llm,
            tool_executor=_make_executor({}),
            terminal_tools=frozenset({"slack_reply"}),
        )
        assert result is not None
        assert result.tool == "slack_reply"
        assert result.args == {"message": "hello"}

    def test_non_agent_skill_also_runs_via_run_agent(self) -> None:
        """agent: false のスキルも run_agent で実行できる（呼び出し元が選択する想定）。"""
        skill = _make_skill("通常スキル本文", agent=False)
        persona = _make_persona()
        chat_input = ChatInput(
            source="test", session_key="s",
            messages=[{"role": "user", "content": "test"}],
            model=None,
        )
        llm = _make_llm(['{"tool": "done", "args": {}}'])
        result = run_agent(
            skill, persona, "", chat_input,
            llm_call=llm,
            tool_executor=_make_executor({}),
            terminal_tools=frozenset({"done"}),
        )
        assert result is not None
        assert result.tool == "done"

    def test_tools_passed_to_agent_runner(self) -> None:
        """skill.meta.tools が AgentRunner に渡り、runner._tools に保存される。"""
        from mltgnt.agent import AgentRunner

        tools_def = [{"name": "search", "description": "検索"}]
        skill = _make_skill("body", agent=True, tools=tools_def)
        persona = _make_persona()
        chat_input = ChatInput(
            source="test", session_key="s",
            messages=[{"role": "user", "content": "x"}],
            model=None,
        )

        captured: list[AgentRunner] = []

        original_run = AgentRunner.run

        def patched_run(self_runner, prompt: str):
            captured.append(self_runner)
            return original_run(self_runner, prompt)

        AgentRunner.run = patched_run  # type: ignore[method-assign]
        try:
            llm = _make_llm(['{"tool": "done", "args": {}}'])
            run_agent(
                skill, persona, "", chat_input,
                llm_call=llm,
                tool_executor=_make_executor({}),
                terminal_tools=frozenset({"done"}),
            )
        finally:
            AgentRunner.run = original_run  # type: ignore[method-assign]

        assert len(captured) == 1
        assert captured[0]._tools == tools_def

    def test_system_prompt_passed_to_agent_runner(self) -> None:
        """skill の system prompt（ペルソナ + 本文）が AgentRunner.run の prompt に渡る。"""
        skill = _make_skill("スキル指示テキスト", agent=True)
        persona = _make_persona("タチコマ")
        chat_input = ChatInput(
            source="test", session_key="s",
            messages=[{"role": "user", "content": "x"}],
            model=None,
        )
        received_prompts: list[str] = []

        def recording_llm(prompt: str, *, tool_result: str | None = None) -> str | None:
            received_prompts.append(prompt)
            return '{"tool": "done", "args": {}}'

        run_agent(
            skill, persona, "", chat_input,
            llm_call=recording_llm,
            tool_executor=_make_executor({}),
            terminal_tools=frozenset({"done"}),
        )
        assert len(received_prompts) == 1
        assert "タチコマ" in received_prompts[0]
        assert "スキル指示テキスト" in received_prompts[0]
