"""
tests/test_skill/test_matcher.py — matcher.match のユニットテスト。

設計: Issue #124 §8 AC-3
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.skill.matcher import match, match_triggers_only, _DEFAULT_MATCHER_MODEL
from mltgnt.skill.models import SkillMeta


def _meta(name: str, triggers: list[str] | None = None) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=f"{name} description",
        argument_hint="",
        model=None,
        path=Path(f"/fake/skills/{name}/SKILL.md"),
        triggers=triggers or [],
    )


SKILLS = {
    "review": _meta("review"),
    "edit": _meta("edit"),
}


class TestMatch:
    @pytest.mark.asyncio
    async def test_match_with_persona_filter(self) -> None:
        """AC-3-1: /review + persona_skills に review あり → マッチ"""
        result = await match("/review 日記/2026-04-17.md", SKILLS, persona_skills=["review", "edit"])
        assert result is not None
        meta, args = result
        assert meta.name == "review"
        assert args == "日記/2026-04-17.md"

    @pytest.mark.asyncio
    async def test_match_filtered_out_by_persona(self) -> None:
        """AC-3-2: /review だが persona_skills に review なし → None"""
        result = await match("/review 日記/2026-04-17.md", SKILLS, persona_skills=["edit"])
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_skill(self) -> None:
        """AC-3-3: /unknown → None"""
        result = await match("/unknown args", SKILLS, persona_skills=None)
        assert result is None

    @pytest.mark.asyncio
    async def test_plain_message(self) -> None:
        """AC-3-4: 普通のメッセージ → triggers/LLM フォールバック（LLM をモック）"""
        from unittest.mock import AsyncMock, patch
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            result = await match("普通のメッセージ", SKILLS, persona_skills=None)
            assert result is None

    @pytest.mark.asyncio
    async def test_no_arguments(self) -> None:
        """AC-3-5: /review 引数なし → arguments = "" """
        result = await match("/review", SKILLS, persona_skills=["review"])
        assert result is not None
        meta, args = result
        assert meta.name == "review"
        assert args == ""

    @pytest.mark.asyncio
    async def test_multiple_spaces(self) -> None:
        """AC-3-6: /review  a  b  c（複数スペース）→ arguments = "a  b  c" """
        result = await match("/review  a  b  c", SKILLS, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "review"
        assert args == "a  b  c"

    @pytest.mark.asyncio
    async def test_no_persona_filter(self) -> None:
        """persona_skills=None ならフィルタなし"""
        result = await match("/review args", SKILLS, persona_skills=None)
        assert result is not None


class TestMatcherModel:
    @pytest.mark.asyncio
    async def test_model_passed_to_llm(self) -> None:
        """model 引数が _match_by_llm の LLM 呼び出しに渡される"""
        with patch("mltgnt.skill.matcher.llm_call") as mock:
            mock.return_value = MagicMock(ok=True, stdout="none")
            await match("hello", SKILLS, model="custom-model")
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_default_model_when_none(self) -> None:
        """model=None のとき _DEFAULT_MATCHER_MODEL が使われる"""
        with patch("mltgnt.skill.matcher.llm_call") as mock:
            mock.return_value = MagicMock(ok=True, stdout="none")
            await match("hello", SKILLS, model=None)
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs["model"] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_empty_string_model_falls_back_to_default(self) -> None:
        """model="" の空文字はデフォルトにフォールバックする"""
        with patch("mltgnt.skill.matcher.llm_call") as mock:
            mock.return_value = MagicMock(ok=True, stdout="none")
            await match("hello", SKILLS, model="")
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs["model"] == _DEFAULT_MATCHER_MODEL

    @pytest.mark.asyncio
    async def test_default_matcher_model_constant(self) -> None:
        """_DEFAULT_MATCHER_MODEL が期待値を持つ"""
        assert _DEFAULT_MATCHER_MODEL == "claude-haiku-4-5-20251001"


class TestMatchTriggersOnly:
    def test_match_by_trigger_keyword(self) -> None:
        """AC1: トリガーキーワードでマッチする"""
        skills = {"calendar": _meta("calendar", triggers=["予定", "スケジュール"])}
        assert match_triggers_only("予定を教えて", skills) == "calendar"

    def test_no_match(self) -> None:
        """AC1: マッチしない入力は None"""
        skills = {"calendar": _meta("calendar", triggers=["予定", "スケジュール"])}
        assert match_triggers_only("こんにちは", skills) is None

    def test_empty_skills(self) -> None:
        """AC1: 空 skills dict は None"""
        assert match_triggers_only("予定", {}) is None

    def test_empty_triggers_only(self) -> None:
        """AC1: triggers が空のスキルのみなら None"""
        skills = {"calendar": _meta("calendar", triggers=[])}
        assert match_triggers_only("予定", skills) is None

    def test_first_match_wins(self) -> None:
        """AC1: 複数マッチ時は最初の一致を返す"""
        skills = {
            "calendar": _meta("calendar", triggers=["予定"]),
            "schedule": _meta("schedule", triggers=["予定"]),
        }
        assert match_triggers_only("予定を教えて", skills) == "calendar"

    def test_no_persona_filter_parameter(self) -> None:
        """AC2: 全エントリを対象にマッチ（ペルソナフィルタなし）"""
        import inspect

        sig = inspect.signature(match_triggers_only)
        assert "persona_skills" not in sig.parameters

        skills = {
            "calendar": _meta("calendar", triggers=["予定"]),
            "other": _meta("other", triggers=["other"]),
        }
        assert match_triggers_only("予定", skills) == "calendar"
