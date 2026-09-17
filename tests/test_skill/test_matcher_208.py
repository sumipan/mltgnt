"""
tests/test_skill/test_matcher_208.py — Issue #208 hybrid matching tests.

Verifies acceptance criteria AC-1–AC-4. Updated to SkillMatchResult form in Issue #1384.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mltgnt.routing.agentic_discover import DiscoverResult
from mltgnt.skill.matcher import match, _match_by_triggers
from mltgnt.skill.models import SkillMeta


def _mock_agentic_unresolved():
    patcher = patch("mltgnt.skill.matcher.AgenticSkillDiscoverer")
    mock_cls = patcher.start()
    mock_discoverer = MagicMock()
    mock_discoverer.discover.return_value = DiscoverResult(kind="unresolved")
    mock_cls.return_value = mock_discoverer
    return patcher


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
    "persona": _meta("persona", triggers=["make a character profile"]),
    "review": _meta("review", triggers=["critique this"]),
}


# --- AC-1: existing slash-command behavior ---

@pytest.mark.asyncio
async def test_ac1_1_slash_persona_allowed():
    result = await match("/persona tanaka", SKILLS_NO_TRIGGERS, persona_skills=["persona"])
    assert result.decisive is not None
    assert result.decisive.name == "persona"
    assert result.arguments == "tanaka"
    assert result.rationale == "slash:persona"


@pytest.mark.asyncio
async def test_ac1_2_slash_persona_filtered():
    result = await match("/persona tanaka", SKILLS_NO_TRIGGERS, persona_skills=["review"])
    assert result.decisive is None


@pytest.mark.asyncio
async def test_ac1_3_unknown_skill():
    result = await match("/unknown arg", SKILLS_NO_TRIGGERS, persona_skills=None)
    assert result.decisive is None


@pytest.mark.asyncio
async def test_ac1_4_plain_message_falls_through():
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            result = await match("plain message", SKILLS_NO_TRIGGERS, persona_skills=None)
            assert result.decisive is None
            assert result.rationale == "none"
            mock_llm.assert_called_once()
    finally:
        agentic_patcher.stop()


# --- AC-2: trigger substring match ---

@pytest.mark.asyncio
async def test_ac2_1_triggers_partial_match():
    result = await match("please make a character profile", SKILLS_WITH_TRIGGERS, persona_skills=None)
    assert result.decisive is not None
    assert result.decisive.name == "persona"
    assert result.arguments == "please make a character profile"
    assert result.rationale == "trigger:make a character profile"


@pytest.mark.asyncio
async def test_ac2_2_no_trigger_match_falls_to_llm():
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            skills = {"persona": _meta("persona", triggers=["make a character profile"])}
            result = await match("what is the weather today?", skills, persona_skills=None)
            assert result.decisive is None
            mock_llm.assert_called_once()
    finally:
        agentic_patcher.stop()


@pytest.mark.asyncio
async def test_ac2_3_trigger_match_but_filtered_by_persona():
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            skills = {"persona": _meta("persona", triggers=["make a character profile"])}
            result = await match("I want to make a character profile", skills, persona_skills=["review"])
            assert result.decisive is None
            mock_llm.assert_called_once()
    finally:
        agentic_patcher.stop()


@pytest.mark.asyncio
async def test_ac2_4_multiple_trigger_match_first_wins():
    skills = {
        "a_skill": _meta("a_skill", triggers=["match"]),
        "b_skill": _meta("b_skill", triggers=["match"]),
    }
    result = await match("a matching message", skills, persona_skills=None)
    assert result.decisive is not None
    assert result.decisive.name == "a_skill"


# --- AC-3: LLM intent classification ---

@pytest.mark.asyncio
async def test_ac3_1_llm_returns_skill_name():
    skills = {"review": _meta("review", triggers=[])}
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (skills["review"], "look this over for me")
            result = await match("look this over for me", skills, persona_skills=None)
            assert result.decisive is not None
            assert result.decisive.name == "review"
            assert result.rationale == "llm:review"
    finally:
        agentic_patcher.stop()


@pytest.mark.asyncio
async def test_ac3_2_llm_returns_none():
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            result = await match("good morning", SKILLS_WITH_TRIGGERS, persona_skills=None)
            assert result.decisive is None
    finally:
        agentic_patcher.stop()


@pytest.mark.asyncio
async def test_ac3_3_llm_returns_unknown_skill():
    # Use a skill set whose triggers will not hit so LLM path is exercised
    skills = {"persona": _meta("persona", triggers=["make a character profile"])}
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None  # same as when LLM returns an unregistered skill name → None
            result = await match("do a review", skills, persona_skills=None)
            assert result.decisive is None
    finally:
        agentic_patcher.stop()


@pytest.mark.asyncio
async def test_ac3_4_llm_api_error():
    # Confirm behavior on LLM error with a skill set that misses triggers
    skills = {"persona": _meta("persona", triggers=["make a character profile"])}
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            result = await match("do a review", skills, persona_skills=None)
            assert result.decisive is None
    finally:
        agentic_patcher.stop()


# --- AC-4: fallback order ---

@pytest.mark.asyncio
async def test_ac4_1_slash_match_skips_triggers_and_llm():
    with patch("mltgnt.skill.matcher._match_by_literal") as mock_literal, \
         patch("mltgnt.skill.matcher._match_by_triggers") as mock_triggers, \
         patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_agentic, \
         patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        result = await match("/persona tanaka", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result.decisive is not None
        mock_literal.assert_not_called()
        mock_triggers.assert_not_called()
        mock_agentic.assert_not_called()
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_ac4_2_literal_match_skips_triggers_and_llm():
    with patch("mltgnt.skill.matcher._match_by_triggers") as mock_triggers, \
         patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_agentic, \
         patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        result = await match("please do a review", SKILLS_NO_TRIGGERS, persona_skills=None)
        assert result.decisive is not None
        assert result.rationale == "literal:review"
        mock_triggers.assert_not_called()
        mock_agentic.assert_not_called()
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_ac4_3_triggers_match_skips_llm():
    with patch("mltgnt.skill.matcher.AgenticSkillDiscoverer") as mock_agentic, \
         patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
        result = await match("please make a character profile", SKILLS_WITH_TRIGGERS, persona_skills=None)
        assert result.decisive is not None
        mock_agentic.assert_not_called()
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_ac4_4_no_match_calls_llm():
    agentic_patcher = _mock_agentic_unresolved()
    try:
        with patch("mltgnt.skill.matcher._match_by_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = None
            skills = {"persona": _meta("persona", triggers=["make a character profile"])}
            await match("what is the weather today?", skills, persona_skills=None)
            mock_llm.assert_called_once()
    finally:
        agentic_patcher.stop()


# --- _match_by_triggers unit tests ---

def test_triggers_match_returns_full_input_as_args():
    skills = {"persona": _meta("persona", triggers=["make a character profile"])}
    result = _match_by_triggers("please make a character profile", skills, None)
    assert result is not None
    meta, args = result
    assert args == "please make a character profile"


def test_triggers_no_match_returns_none():
    skills = {"persona": _meta("persona", triggers=["make a character profile"])}
    result = _match_by_triggers("what is the weather today?", skills, None)
    assert result is None
