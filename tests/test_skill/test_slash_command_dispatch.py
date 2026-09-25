"""
tests/test_skill/test_slash_command_dispatch.py — #208 isolation tests.

Reproduce and isolate the issue where /persona-create does not enter the skill execution path.
Hypotheses:
  1) resolve_skill() does not match at all (name mismatch)
  2) it matches but does not enter the execution path (routing side)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mltgnt.routing.agentic_discover import DiscoverResult
from mltgnt.skill.loader import discover, load
from mltgnt.skill.matcher import match
from mltgnt.skill.models import SkillMatchResult, SkillMeta
from mltgnt.skill import resolve_skill


def _mock_agentic_unresolved():
    patcher = patch("mltgnt.skill.matcher.AgenticSkillDiscoverer")
    mock_cls = patcher.start()
    mock_discoverer = MagicMock()
    mock_discoverer.discover.return_value = DiscoverResult(kind="unresolved")
    mock_cls.return_value = mock_discoverer
    return patcher


# --------------- matcher unit tests (hypothesis 1) ---------------

def _meta(name: str) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=f"{name} description",
        argument_hint="",
        model=None,
        path=Path(f"/fake/skills/{name}/SKILL.md"),
    )


SKILLS_WITH_HYPHEN = {
    "review": _meta("review"),
    "persona-create": _meta("persona-create"),
    "diary-review-sakuma": _meta("diary-review-sakuma"),
}


class TestHyphenatedSlashCommand:
    """match() tests for hyphenated skill names."""

    async def test_persona_create_matches(self) -> None:
        """1) Does /persona-create match at all?"""
        result = await match("/persona-create Fumio Koga", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "persona-create"
        assert result.arguments == "Fumio Koga"

    async def test_persona_create_no_args(self) -> None:
        """/persona-create matches even with no args"""
        result = await match("/persona-create", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "persona-create"
        assert result.arguments == ""

    async def test_triple_hyphen_name(self) -> None:
        """Also matches three-segment hyphenated names"""
        result = await match("/diary-review-sakuma", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "diary-review-sakuma"

    async def test_persona_create_filtered_by_persona_skills(self) -> None:
        """decisive=None when not included in persona_skills"""
        result = await match(
            "/persona-create Fumio Koga",
            SKILLS_WITH_HYPHEN,
            persona_skills=["review"],
        )
        assert result.decisive is None

    async def test_persona_create_allowed_by_persona_skills(self) -> None:
        """matches when included in persona_skills"""
        result = await match(
            "/persona-create Fumio Koga",
            SKILLS_WITH_HYPHEN,
            persona_skills=["persona-create"],
        )
        assert result.decisive is not None
        assert result.decisive.name == "persona-create"


# --------------- Slack-style input preprocessing tests ---------------

class TestSlackInputEdgeCases:
    """Ensure Slack-specific input patterns do not break matching.
    Mock the LLM fallback and verify slash-pattern behavior alone.
    """

    async def test_leading_whitespace(self) -> None:
        """Leading space misses slash pattern but literal matcher picks it up"""
        agentic_patcher = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
                result = await match(" /persona-create Fumio Koga", SKILLS_WITH_HYPHEN, persona_skills=None)
            assert result.decisive is not None
            assert result.decisive.name == "persona-create"
            assert result.rationale.startswith("literal:")
        finally:
            agentic_patcher.stop()

    async def test_leading_newline(self) -> None:
        """Leading newline misses slash pattern but literal matcher picks it up"""
        agentic_patcher = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
                result = await match("\n/persona-create Fumio Koga", SKILLS_WITH_HYPHEN, persona_skills=None)
            assert result.decisive is not None
            assert result.decisive.name == "persona-create"
            assert result.rationale.startswith("literal:")
        finally:
            agentic_patcher.stop()

    async def test_multiline_with_slash_on_first_line(self) -> None:
        """/name on the first line with following newline text"""
        result = await match("/persona-create Fumio Koga\nNice to meet you", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "persona-create"
        assert "Fumio Koga" in result.arguments
        assert "Nice to meet you" in result.arguments

    async def test_slash_in_middle_of_text(self) -> None:
        """/name mid-text misses slash pattern but literal matcher picks it up"""
        agentic_patcher = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
                result = await match("I want to use /persona-create today", SKILLS_WITH_HYPHEN, persona_skills=None)
            assert result.decisive is not None
            assert result.decisive.name == "persona-create"
            assert result.rationale.startswith("literal:")
        finally:
            agentic_patcher.stop()


# --------------- discover + match integration (hypothesis 1 filesystem) ---------------

PERSONA_CREATE_SKILL_MD = """\
---
name: persona-create
description: >
  Auto-generate a persona file for a person.
argument_hint: "<person name>"
model: null
---

body here
"""


def _write_skill(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestDiscoverAndMatchIntegration:
    """Integration test: pass discover() skills dict into match()."""

    def test_discover_finds_hyphenated_skill(self, tmp_path: Path) -> None:
        """SKILL.md under persona-create directory is discovered"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        assert "persona-create" in skills

    async def test_discover_then_match(self, tmp_path: Path) -> None:
        """/persona-create matches through discover → match pipeline"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        result = await match("/persona-create Fumio Koga", skills, persona_skills=None)
        assert result.decisive is not None
        assert result.decisive.name == "persona-create"
        assert result.arguments == "Fumio Koga"

    async def test_discover_then_match_then_load(self, tmp_path: Path) -> None:
        """Full discover → match → load pipeline"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        result = await match("/persona-create Fumio Koga", skills, persona_skills=None)
        assert result.decisive is not None
        skill_file = load(result.decisive)
        assert skill_file.meta.name == "persona-create"
        assert "body here" in skill_file.body


# --------------- resolve_skill integration (hypothesis 2 routing) ---------------

class TestResolveSkillIntegration:
    """resolve_skill() integration test. Resolves from the filesystem."""

    async def test_resolve_persona_create(self, tmp_path: Path) -> None:
        """resolve_skill can resolve /persona-create"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill("/persona-create Fumio Koga", [tmp_path])
        assert result is not None
        skill_file, args = result
        assert skill_file.meta.name == "persona-create"
        assert args == "Fumio Koga"

    async def test_resolve_persona_create_with_persona_filter_pass(self, tmp_path: Path) -> None:
        """resolves when included in persona_skills"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill(
            "/persona-create Fumio Koga",
            [tmp_path],
            persona_skills=["persona-create"],
        )
        assert result is not None

    async def test_resolve_persona_create_with_persona_filter_block(self, tmp_path: Path) -> None:
        """returns None when not in persona_skills"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill(
            "/persona-create Fumio Koga",
            [tmp_path],
            persona_skills=["review"],
        )
        assert result is None

    async def test_resolve_plain_text_returns_none(self, tmp_path: Path) -> None:
        """Plain text without slash returns None when LLM is mocked (no slash/trigger match)"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        agentic_patcher = _mock_agentic_unresolved()
        try:
            with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
                result = await resolve_skill("create a persona", [tmp_path])
            assert result is None
        finally:
            agentic_patcher.stop()

    async def test_resolve_with_empty_paths(self) -> None:
        """empty path list → None"""
        result = await resolve_skill("/persona-create foo", [])
        assert result is None

    async def test_resolve_skill_passes_matcher_model(self, tmp_path: Path) -> None:
        """matcher_model='custom' is passed to match() as model='custom'"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        no_match = SkillMatchResult(decisive=None, candidates=[], rationale="none", arguments="hello")
        with patch("mltgnt.skill.match", new=AsyncMock(return_value=no_match)) as mock_match:
            await resolve_skill("hello", [tmp_path], matcher_model="custom-model")
            mock_match.assert_called_once()
            _, kwargs = mock_match.call_args
            assert kwargs.get("model") == "custom-model"

    async def test_resolve_skill_passes_matcher_engine(self, tmp_path: Path) -> None:
        """matcher_engine='codex' is passed to match() as engine='codex'"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        no_match = SkillMatchResult(decisive=None, candidates=[], rationale="none", arguments="hello")
        with patch("mltgnt.skill.match", new=AsyncMock(return_value=no_match)) as mock_match:
            await resolve_skill("hello", [tmp_path], matcher_engine="codex")
            mock_match.assert_called_once()
            _, kwargs = mock_match.call_args
            assert kwargs.get("engine") == "codex"

    async def test_resolve_skill_default_matcher_engine(self, tmp_path: Path) -> None:
        """matcher_engine unset -> match() receives engine='claude'"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        no_match = SkillMatchResult(decisive=None, candidates=[], rationale="none", arguments="hello")
        with patch("mltgnt.skill.match", new=AsyncMock(return_value=no_match)) as mock_match:
            await resolve_skill("hello", [tmp_path])
            mock_match.assert_called_once()
            _, kwargs = mock_match.call_args
            assert kwargs.get("engine") == "claude"

    async def test_resolve_with_real_skills_dir(self) -> None:
        """Resolve from the real skills/ directory (only when SKILL.md exists)"""
        real_skills_dir = Path("/Users/ngystks/Github/diary/skills")
        skill_md = real_skills_dir / "persona-create" / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("skills/persona-create/SKILL.md does not exist")
        result = await resolve_skill("/persona-create test person", [real_skills_dir])
        assert result is not None, (
            "skills/persona-create/SKILL.md exists but "
            "resolve_skill returned None — problem in discover or match"
        )
        skill_file, args = result
        assert skill_file.meta.name == "persona-create"
        assert args == "test person"
