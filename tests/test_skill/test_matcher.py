"""
tests/test_skill/test_matcher.py — matcher.match のユニットテスト。

設計: Issue #124 §8 AC-3
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.skill.matcher import match
from mltgnt.skill.models import SkillMeta


def _meta(name: str) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=f"{name} description",
        argument_hint="",
        model=None,
        path=Path(f"/fake/skills/{name}/SKILL.md"),
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
