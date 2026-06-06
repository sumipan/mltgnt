"""
tests/test_skill/test_matcher.py — matcher.match のユニットテスト。

設計: Issue #124 §8 AC-3, Issue #1384 U5
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mltgnt.skill.agentic_discover import DiscoverResult
from mltgnt.skill.matcher import match, match_pipeline, match_triggers_only, split_pipe_segments, _DEFAULT_MATCHER_MODEL
from mltgnt.skill.models import SkillMeta


def _mock_agentic_unresolved():
    """AgenticSkillDiscoverer を unresolved に固定するパッチ。"""
    patcher = patch("mltgnt.skill.matcher.AgenticSkillDiscoverer")
    mock_cls = patcher.start()
    mock_discoverer = MagicMock()
    mock_discoverer.discover.return_value = DiscoverResult(kind="unresolved")
    mock_cls.return_value = mock_discoverer
    return patcher, mock_cls, mock_discoverer


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
        assert result.decisive is not None
        assert result.decisive.name == "review"
        assert result.arguments == "日記/2026-04-17.md"
        assert result.rationale == "slash:review"

    @pytest.mark.asyncio
    async def test_match_filtered_out_by_persona(self) -> None:
        """AC-3-2: /review だが persona_skills に review なし → decisive=None"""
        result = await match("/review 日記/2026-04-17.md", SKILLS, persona_skills=["edit"])
        assert result.decisive is None
        assert result.rationale == "none"

    @pytest.mark.asyncio
    async def test_unknown_skill(self) -> None:
        """AC-3-3: /unknown → decisive=None"""
        result = await match("/unknown args", SKILLS, persona_skills=None)
        assert result.decisive is None
        assert result.rationale == "none"

    @pytest.mark.asyncio
    async def test_plain_message(self) -> None:
        """AC-3-4: 普通のメッセージ → triggers/LLM フォールバック（LLM をモック）"""
        agentic_patcher, _, _ = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = None
                result = await match("普通のメッセージ", SKILLS, persona_skills=None)
                assert result.decisive is None
                assert result.rationale == "none"
        finally:
            agentic_patcher.stop()

    @pytest.mark.asyncio
    async def test_no_arguments(self) -> None:
        """AC-3-5: /review 引数なし → arguments = "" """
        result = await match("/review", SKILLS, persona_skills=["review"])
        assert result.decisive is not None
        assert result.decisive.name == "review"
        assert result.arguments == ""
        assert result.rationale == "slash:review"

    @pytest.mark.asyncio
    async def test_multiple_spaces(self) -> None:
        """AC-3-6: /review  a  b  c（複数スペース）→ arguments = "a  b  c" """
        result = await match("/review  a  b  c", SKILLS, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "review"
        assert result.arguments == "a  b  c"

    @pytest.mark.asyncio
    async def test_no_persona_filter(self) -> None:
        """persona_skills=None ならフィルタなし"""
        result = await match("/review args", SKILLS, persona_skills=None)
        assert result.decisive is not None

    @pytest.mark.asyncio
    async def test_literal_match(self) -> None:
        """AC1: リテラル名一致 → rationale=literal:<name>"""
        result = await match("reviewをお願い", SKILLS, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "review"
        assert result.arguments == "reviewをお願い"
        assert result.rationale == "literal:review"

    @pytest.mark.asyncio
    async def test_literal_multiple_hits_falls_through(self) -> None:
        """AC1: 複数リテラルヒット → triggers/LLM にフォールバック"""
        agentic_patcher, _, _ = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = None
                result = await match("reviewとeditの両方", SKILLS, persona_skills=None)
                assert result.decisive is None
                mock_llm.assert_called_once()
        finally:
            agentic_patcher.stop()


class TestMatcherModel:
    @pytest.mark.asyncio
    async def test_model_passed_to_llm(self) -> None:
        """model 引数が _match_by_llm の LLM 呼び出しに渡される"""
        agentic_patcher, _, _ = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher.llm_call") as mock:
                mock.return_value = MagicMock(ok=True, stdout="none")
                await match("hello", SKILLS, model="custom-model")
                mock.assert_called_once()
                _, kwargs = mock.call_args
                assert kwargs["model"] == "custom-model"
        finally:
            agentic_patcher.stop()

    @pytest.mark.asyncio
    async def test_default_model_when_none(self) -> None:
        """model=None のとき _DEFAULT_MATCHER_MODEL が使われる"""
        agentic_patcher, _, _ = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher.llm_call") as mock:
                mock.return_value = MagicMock(ok=True, stdout="none")
                await match("hello", SKILLS, model=None)
                mock.assert_called_once()
                _, kwargs = mock.call_args
                assert kwargs["model"] == "claude-haiku-4-5-20251001"
        finally:
            agentic_patcher.stop()

    @pytest.mark.asyncio
    async def test_empty_string_model_falls_back_to_default(self) -> None:
        """model="" の空文字はデフォルトにフォールバックする"""
        agentic_patcher, _, _ = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher.llm_call") as mock:
                mock.return_value = MagicMock(ok=True, stdout="none")
                await match("hello", SKILLS, model="")
                mock.assert_called_once()
                _, kwargs = mock.call_args
                assert kwargs["model"] == _DEFAULT_MATCHER_MODEL
        finally:
            agentic_patcher.stop()

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


class TestAgenticDiscover:
    CALENDAR_SKILLS = {
        "calendar": _meta("calendar", triggers=[]),
        "diary": _meta("diary", triggers=[]),
    }

    @pytest.mark.asyncio
    async def test_agentic_selected(self) -> None:
        calendar_meta = self.CALENDAR_SKILLS["calendar"]
        with patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_cls:
            mock_discoverer = MagicMock()
            mock_discoverer.discover.return_value = DiscoverResult(
                kind="selected", skill=calendar_meta
            )
            mock_cls.return_value = mock_discoverer
            result = await match("予定を教えて", self.CALENDAR_SKILLS, persona_skills=None)
            assert result.decisive == calendar_meta
            assert result.rationale == "agentic:calendar"

    @pytest.mark.asyncio
    async def test_agentic_ambiguous(self) -> None:
        cal = self.CALENDAR_SKILLS["calendar"]
        diary = self.CALENDAR_SKILLS["diary"]
        with patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_cls:
            mock_discoverer = MagicMock()
            mock_discoverer.discover.return_value = DiscoverResult(
                kind="ambiguous",
                candidates=[(cal, 0.8), (diary, 0.6)],
            )
            mock_cls.return_value = mock_discoverer
            result = await match("予定を教えて", self.CALENDAR_SKILLS, persona_skills=None)
            assert result.decisive == cal
            assert result.rationale == "agentic-ambiguous:calendar"

    @pytest.mark.asyncio
    async def test_agentic_unresolved_falls_to_llm(self) -> None:
        review_meta = SKILLS["review"]
        with patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_cls, \
             patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_discoverer = MagicMock()
            mock_discoverer.discover.return_value = DiscoverResult(kind="unresolved")
            mock_cls.return_value = mock_discoverer
            mock_llm.return_value = (review_meta, "レビューお願い")
            result = await match("レビューお願い", SKILLS, persona_skills=None)
            mock_llm.assert_called_once()
            assert result.decisive == review_meta
            assert result.rationale == "llm:review"

    @pytest.mark.asyncio
    async def test_agentic_exception_falls_to_llm(self) -> None:
        with patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_cls, \
             patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm, \
             patch("mltgnt.skill.matcher._log") as mock_log:
            mock_discoverer = MagicMock()
            mock_discoverer.discover.side_effect = RuntimeError("LLM down")
            mock_cls.return_value = mock_discoverer
            mock_llm.return_value = None
            result = await match("hello", SKILLS, persona_skills=None)
            mock_llm.assert_called_once()
            mock_log.warning.assert_called_once()
            assert result.rationale == "none"


class TestSplitPipeSegments:
    def test_two_segments(self) -> None:
        assert split_pipe_segments("/skill-a foo | /skill-b") == ["/skill-a foo", "/skill-b"]

    def test_three_segments(self) -> None:
        assert split_pipe_segments("/skill-a foo | /skill-b bar | /skill-c") == [
            "/skill-a foo",
            "/skill-b bar",
            "/skill-c",
        ]

    def test_no_spaces_around_pipe(self) -> None:
        assert split_pipe_segments("/skill-a foo|bar") == ["/skill-a foo|bar"]

    def test_no_pipe(self) -> None:
        assert split_pipe_segments("plain text") == ["plain text"]


class TestMatchPipeline:
    PIPELINE_SKILLS = {
        "skill-a": _meta("skill-a"),
        "skill-b": _meta("skill-b"),
        "skill-c": _meta("skill-c"),
    }

    @pytest.mark.asyncio
    async def test_two_skill_pipeline(self) -> None:
        results = await match_pipeline("/skill-a foo | /skill-b", self.PIPELINE_SKILLS)
        assert len(results) == 2
        assert results[0].decisive is not None
        assert results[0].decisive.name == "skill-a"
        assert results[0].arguments == "foo"
        assert results[1].decisive is not None
        assert results[1].decisive.name == "skill-b"
        assert results[1].arguments == ""

    @pytest.mark.asyncio
    async def test_single_skill_backward_compat(self) -> None:
        results = await match_pipeline("/single-skill arg", {"single-skill": _meta("single-skill")})
        assert len(results) == 1
        assert results[0].decisive is not None
        assert results[0].decisive.name == "single-skill"
        assert results[0].arguments == "arg"

    @pytest.mark.asyncio
    async def test_trigger_pipeline(self) -> None:
        skills = {
            "skill-a": _meta("skill-a", triggers=["trigger-a"]),
            "skill-b": _meta("skill-b", triggers=["trigger-b"]),
        }
        results = await match_pipeline("trigger-a | trigger-b", skills)
        assert len(results) == 2
        assert results[0].decisive is not None
        assert results[0].decisive.name == "skill-a"
        assert results[1].decisive is not None
        assert results[1].decisive.name == "skill-b"
