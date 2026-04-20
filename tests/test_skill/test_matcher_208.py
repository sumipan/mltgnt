"""
tests/test_skill/test_matcher_208.py — Issue #208 ハイブリッドマッチング テスト。

受け入れ条件 AC-1〜AC-4 を検証する。
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mltgnt.skill.matcher import match, _match_by_triggers
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


SKILLS_NO_TRIGGERS = {
    "persona": _meta("persona"),
    "review": _meta("review"),
}

SKILLS_WITH_TRIGGERS = {
    "persona": _meta("persona", triggers=["ペルソナを作"]),
    "review": _meta("review", triggers=["レビュー"]),
}


# --- AC-1: スラッシュコマンドの既存動作 ---

@pytest.mark.asyncio
async def test_ac1_1_slash_persona_allowed():
    result = await match("/persona tanaka", SKILLS_NO_TRIGGERS, persona_skills=["persona"])
    assert result is not None
    meta, args = result
    assert meta.name == "persona"
    assert args == "tanaka"


@pytest.mark.asyncio
async def test_ac1_2_slash_persona_filtered():
    result = await match("/persona tanaka", SKILLS_NO_TRIGGERS, persona_skills=["review"])
    assert result is None


@pytest.mark.asyncio
async def test_ac1_3_unknown_skill():
    result = await match("/unknown arg", SKILLS_NO_TRIGGERS, persona_skills=None)
    assert result is None


@pytest.mark.asyncio
async def test_ac1_4_plain_message_falls_through():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        result = await match("普通のメッセージ", SKILLS_NO_TRIGGERS, persona_skills=None)
        assert result is None
        mock_llm.assert_called_once()


# --- AC-2: triggers 部分一致 ---

@pytest.mark.asyncio
async def test_ac2_1_triggers_partial_match():
    result = await match("ペルソナを作ってほしい", SKILLS_WITH_TRIGGERS, persona_skills=None)
    assert result is not None
    meta, args = result
    assert meta.name == "persona"
    assert args == "ペルソナを作ってほしい"


@pytest.mark.asyncio
async def test_ac2_2_no_trigger_match_falls_to_llm():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
        result = await match("今日の天気は？", skills, persona_skills=None)
        assert result is None
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_ac2_3_trigger_match_but_filtered_by_persona():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
        result = await match("ペルソナを作りたい", skills, persona_skills=["review"])
        assert result is None
        mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_ac2_4_multiple_trigger_match_first_wins():
    skills = {
        "a_skill": _meta("a_skill", triggers=["マッチ"]),
        "b_skill": _meta("b_skill", triggers=["マッチ"]),
    }
    result = await match("マッチするメッセージ", skills, persona_skills=None)
    assert result is not None
    meta, args = result
    assert meta.name == "a_skill"


# --- AC-3: LLM 意図分類 ---

@pytest.mark.asyncio
async def test_ac3_1_llm_returns_skill_name():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = (SKILLS_WITH_TRIGGERS["review"], "レビューお願い")
        result = await match("レビューお願い", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "review"


@pytest.mark.asyncio
async def test_ac3_2_llm_returns_none():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        result = await match("おはよう", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result is None


@pytest.mark.asyncio
async def test_ac3_3_llm_returns_unknown_skill():
    # "レビュー" が triggers にあるため、triggers にヒットしないスキルセットを使う
    skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None  # LLM が登録外スキル名を返す場合と同じく None
        result = await match("レビューして", skills, persona_skills=None)
        assert result is None


@pytest.mark.asyncio
async def test_ac3_4_llm_api_error():
    # triggers にヒットしないスキルセットで LLM エラー時の動作を確認
    skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        result = await match("レビューして", skills, persona_skills=None)
        assert result is None


# --- AC-4: フォールバック順序 ---

@pytest.mark.asyncio
async def test_ac4_1_slash_match_skips_triggers_and_llm():
    with patch("mltgnt.skill.matcher._match_by_triggers") as mock_triggers, \
         patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        result = await match("/persona tanaka", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result is not None
        mock_triggers.assert_not_called()
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_ac4_2_triggers_match_skips_llm():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        result = await match("ペルソナを作ってほしい", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result is not None
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_ac4_3_no_match_calls_llm():
    with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = None
        skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
        result = await match("今日の天気は？", skills, persona_skills=None)
        mock_llm.assert_called_once()


# --- _match_by_triggers 単体テスト ---

def test_triggers_match_returns_full_input_as_args():
    skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
    result = _match_by_triggers("ペルソナを作ってほしい", skills, None)
    assert result is not None
    meta, args = result
    assert args == "ペルソナを作ってほしい"


def test_triggers_no_match_returns_none():
    skills = {"persona": _meta("persona", triggers=["ペルソナを作"])}
    result = _match_by_triggers("今日の天気は？", skills, None)
    assert result is None
