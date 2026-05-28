from __future__ import annotations

from pathlib import Path

from mltgnt.improvement import FailurePattern, generate_proposals


def _pattern(
    pattern_id: str,
    category: str,
    count: int,
    *,
    persona: str | None = None,
    skill: str | None = None,
) -> FailurePattern:
    return FailurePattern(
        pattern_id=pattern_id,
        category=category,
        count=count,
        example_correlation_ids=[f"{pattern_id}-corr"],
        affected_persona=persona,
        affected_skill=skill,
    )


def _touch(path: Path, text: str = "stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_generate_proposals_returns_empty_for_low_count(tmp_path):
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    _touch(persona_dir / "タチコマ.md")
    _touch(skills_dir / "system-improve-agents" / "SKILL.md")

    patterns = [_pattern("p1", "triage_error", 2, persona="タチコマ")]
    proposals = generate_proposals(patterns, persona_dir, skills_dir, min_count=3)
    assert proposals == []


def test_generate_proposals_creates_rule_based_proposals(tmp_path):
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    _touch(persona_dir / "タチコマ.md")
    _touch(skills_dir / "system-improve-agents" / "SKILL.md")

    patterns = [
        _pattern("triage:タチコマ", "triage_error", 4, persona="タチコマ"),
        _pattern("skill:system-improve-agents", "skill_mismatch", 5, skill="system-improve-agents"),
        _pattern("timeout:タチコマ", "timeout", 3, persona="タチコマ"),
        _pattern("quality:タチコマ", "persona_quality", 3, persona="タチコマ"),
    ]

    proposals = generate_proposals(patterns, persona_dir, skills_dir)
    assert len(proposals) == 4
    by_action = {proposal.action: proposal for proposal in proposals}
    assert set(by_action) == {"adjust_section", "add_trigger", "add_rule"}
    assert all(proposal.diff_preview for proposal in proposals)
    assert all(proposal.confidence >= 0.5 for proposal in proposals)


def test_generate_proposals_skips_unknown_target(tmp_path):
    persona_dir = tmp_path / "personas"
    skills_dir = tmp_path / "skills"
    _touch(persona_dir / "既知.md")
    _touch(skills_dir / "known-skill" / "SKILL.md")

    patterns = [
        _pattern("unknown-persona", "triage_error", 3, persona="未知"),
        _pattern("known-skill", "skill_mismatch", 3, skill="known-skill"),
    ]

    proposals = generate_proposals(patterns, persona_dir, skills_dir)
    assert len(proposals) == 1
    assert proposals[0].target_name == "known-skill"
