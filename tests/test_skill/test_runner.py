"""
tests/test_skill/test_runner.py — unit tests for runner.run.

Design: Issue #124 §8 AC-4, AC-5, Issue #1384 U6
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mltgnt.interfaces.types import ChatInput
from mltgnt.skill.models import ProducesSpec, SkillFile, SkillMeta, SkillRunResult
from mltgnt.skill.runner import run


def _make_skill(
    body: str,
    model: str | None = None,
    name: str = "review",
    *,
    skill_io: str = "legacy",
    produces: ProducesSpec | None = None,
) -> SkillFile:
    meta = SkillMeta(
        name=name,
        description="test",
        argument_hint="",
        model=model,
        path=Path("/fake/skills/review/SKILL.md"),
        skill_io=skill_io,
        produces=produces,
    )
    return SkillFile(meta=meta, body=body)


def _make_persona(name: str = "persona-a") -> MagicMock:
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
        """AC-4-1: $ARGUMENTS, $0, $1 substitution"""
        skill = _make_skill("file=$0 mode=$1 all=$ARGUMENTS")
        persona = _make_persona()
        result = run(skill, persona, "diary/2026-04-17.md critique", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "diary/2026-04-17.md critique" in sys_content  # $ARGUMENTS
        assert "file=diary/2026-04-17.md" in sys_content       # $0
        assert "mode=critique" in sys_content                  # $1

    def test_empty_arguments(self) -> None:
        """AC-4-2: empty args → $ARGUMENTS → "", $0 → """
        skill = _make_skill("args=[$ARGUMENTS] pos=[$0]")
        persona = _make_persona()
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "args=[]" in sys_content
        assert "pos=[]" in sys_content

    def test_persona_substitution(self) -> None:
        """AC-4-3: $PERSONA → persona name"""
        skill = _make_skill("persona=$PERSONA")
        persona = _make_persona("persona-a")
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "persona-a" in sys_content

    def test_skill_dir_substitution(self) -> None:
        """AC-4-4: $SKILL_DIR → parent directory of SKILL.md"""
        skill = _make_skill("dir=$SKILL_DIR")
        persona = _make_persona()
        result = run(skill, persona, "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "/fake/skills/review" in sys_content

    def test_out_of_range_positional(self) -> None:
        """AC-4-5: $3 with only 2 args → empty string"""
        skill = _make_skill("$3")
        persona = _make_persona()
        result = run(skill, persona, "a b", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "$3" not in sys_content  # substituted
        # $3 → "" so that part is empty
        assert "[PERSONA:" in sys_content


class TestRunPromptComposition:
    def test_system_prompt_contains_persona_and_skill(self) -> None:
        """AC-5-1: system prompt includes persona instructions and skill body"""
        skill = _make_skill("skill body")
        persona = _make_persona("persona-a")
        result = run(skill, persona, "", _make_chat_input())
        assert result.chat_input.messages[0]["role"] == "system"
        sys_content = result.chat_input.messages[0]["content"]
        assert "persona-a" in sys_content
        assert "skill body" in sys_content

    def test_skill_model_overrides(self) -> None:
        """AC-5-2: when skill.meta.model is set, returned model uses skill value"""
        skill = _make_skill("body", model="claude-opus-4-6")
        result = run(skill, _make_persona(), "", _make_chat_input(model="default-model"))
        assert result.chat_input.model == "claude-opus-4-6"

    def test_null_model_inherits(self) -> None:
        """AC-5-3: when skill.meta.model is null, inherit chat_input.model"""
        skill = _make_skill("body", model=None)
        result = run(skill, _make_persona(), "", _make_chat_input(model="default-model"))
        assert result.chat_input.model == "default-model"

    def test_user_message_preserved(self) -> None:
        """Original user message is preserved"""
        skill = _make_skill("body")
        chat_input = _make_chat_input()
        result = run(skill, _make_persona(), "", chat_input)
        user_msgs = [m for m in result.chat_input.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "hello"

    def test_original_system_message_replaced(self) -> None:
        """Original system message is replaced by the new system prompt"""
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
        system_msgs = [m for m in result.chat_input.messages if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "old system" not in system_msgs[0]["content"]
        assert "new system" in system_msgs[0]["content"]


class TestSkillRunResult:
    def test_returns_skill_run_result_instance(self) -> None:
        result = run(_make_skill("body"), _make_persona(), "", _make_chat_input())
        assert isinstance(result, SkillRunResult)

    def test_diagnostics_defaults_to_empty_list(self) -> None:
        result = run(_make_skill("body"), _make_persona(), "", _make_chat_input())
        assert result.diagnostics == []

    def test_expected_markers_from_produces(self) -> None:
        produces = ProducesSpec(status_markers=["DONE", "ERROR"])
        skill = _make_skill("body", produces=produces, skill_io="v1")
        result = run(skill, _make_persona(), "", _make_chat_input())
        assert result.expected_markers == ["DONE", "ERROR"]
        assert result.skill_io == "v1"

    def test_expected_markers_empty_when_no_produces(self) -> None:
        skill = _make_skill("body", produces=None, skill_io="legacy")
        result = run(skill, _make_persona(), "", _make_chat_input())
        assert result.expected_markers == []
        assert result.skill_io == "legacy"


class TestRunEnvVarSubstitution:
    def test_nikki_root_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NIKKI_ROOT", "/path/to/diary")
        skill = _make_skill("root=$NIKKI_ROOT")
        result = run(skill, _make_persona(), "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "/path/to/diary" in sys_content

    def test_repo_root_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REPO_ROOT", "/path/to/nexus")
        skill = _make_skill("root=$REPO_ROOT")
        result = run(skill, _make_persona(), "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "/path/to/nexus" in sys_content

    def test_nikki_root_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NIKKI_ROOT", raising=False)
        skill = _make_skill("root=[$NIKKI_ROOT]")
        result = run(skill, _make_persona(), "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "root=[]" in sys_content

    def test_repo_root_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("REPO_ROOT", raising=False)
        skill = _make_skill("root=[$REPO_ROOT]")
        result = run(skill, _make_persona(), "", _make_chat_input())
        sys_content = result.chat_input.messages[0]["content"]
        assert "root=[]" in sys_content


class TestRunExtraContext:
    """Issue #3021: context injection via extra_context."""

    def test_extra_context_appends_section(self) -> None:
        """When extra_context is passed, the production context header is inserted into the system prompt."""
        skill = _make_skill("skill body")
        result = run(
            skill,
            _make_persona(),
            "",
            _make_chat_input(),
            extra_context="knowledge and memory snippet",
        )
        sys_content = result.chat_input.messages[0]["content"]
        assert "skill body" in sys_content
        # Japanese text intentionally kept for CJK processing test
        assert "## Context" in sys_content
        assert "knowledge and memory snippet" in sys_content
        # Context is appended after the skill body
        assert sys_content.index("skill body") < sys_content.index("## Context")

    def test_extra_context_none_is_backward_compatible(self) -> None:
        """Without extra_context / None, output matches prior behavior."""
        skill = _make_skill("skill body")
        persona = _make_persona()
        chat_input = _make_chat_input()
        without = run(skill, persona, "", chat_input)
        with_none = run(skill, persona, "", chat_input, extra_context=None)
        assert without.chat_input.messages[0]["content"] == with_none.chat_input.messages[0]["content"]
        # Japanese text intentionally kept for CJK processing test
        assert "## Context" not in without.chat_input.messages[0]["content"]


class TestWriteResultFrontmatter:
    def test_v1_writes_frontmatter(self, tmp_path: Path) -> None:
        """AC-3: skill_io=v1 skill writes frontmatter into the result file"""
        from mltgnt.skill.runner import write_result_frontmatter

        result_path = tmp_path / "result.md"
        result_path.write_text("body content\n", encoding="utf-8")
        produces = ProducesSpec(
            content_type="text/markdown",
            status_markers=["ACCEPTED"],
        )
        run_result = SkillRunResult(
            chat_input=_make_chat_input(),
            expected_markers=["ACCEPTED"],
            skill_io="v1",
            produces=produces,
        )
        write_result_frontmatter(result_path, run_result)
        text = result_path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "skill_io: v1" in text
        assert "content_type: text/markdown" in text
        assert "ACCEPTED" in text
        assert "body content" in text

    def test_legacy_noop(self, tmp_path: Path) -> None:
        """AC-3: skill_io=legacy skill is a noop"""
        from mltgnt.skill.runner import write_result_frontmatter

        result_path = tmp_path / "result.md"
        original = "legacy body\n"
        result_path.write_text(original, encoding="utf-8")
        run_result = SkillRunResult(
            chat_input=_make_chat_input(),
            expected_markers=[],
            skill_io="legacy",
            produces=ProducesSpec(content_type="text/markdown"),
        )
        write_result_frontmatter(result_path, run_result)
        assert result_path.read_text(encoding="utf-8") == original

    def test_produces_none_noop(self, tmp_path: Path) -> None:
        """AC-3: produces=None is a noop"""
        from mltgnt.skill.runner import write_result_frontmatter

        result_path = tmp_path / "result.md"
        original = "no produces\n"
        result_path.write_text(original, encoding="utf-8")
        run_result = SkillRunResult(
            chat_input=_make_chat_input(),
            expected_markers=[],
            skill_io="v1",
            produces=None,
        )
        write_result_frontmatter(result_path, run_result)
        assert result_path.read_text(encoding="utf-8") == original

    def test_run_sets_produces(self) -> None:
        """AC-3: run() return value includes produces"""
        produces = ProducesSpec(content_type="text/plain", status_markers=["DONE"])
        skill = _make_skill("body", skill_io="v1", produces=produces)
        result = run(skill, _make_persona(), "", _make_chat_input())
        assert result.produces is produces
        assert result.skill_io == "v1"
