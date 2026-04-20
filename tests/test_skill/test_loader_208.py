"""
tests/test_skill/test_loader_208.py — Issue #208 AC-5: triggers パース テスト。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.skill.loader import _build_meta, _parse_frontmatter


def _fm_and_meta(yaml_content: str, path: Path | None = None) -> object:
    text = f"---\n{yaml_content}\n---\nbody\n"
    fm, _ = _parse_frontmatter(text)
    return _build_meta(fm, path or Path("/fake/skills/test/SKILL.md"))


def test_ac5_1_triggers_list():
    """AC-5-1: triggers: ["a", "b"] -> SkillMeta.triggers == ["a", "b"]"""
    meta = _fm_and_meta('name: test\ndescription: desc\ntriggers:\n  - "a"\n  - "b"\n')
    assert meta.triggers == ["a", "b"]


def test_ac5_2_triggers_missing():
    """AC-5-2: triggers キー未指定 -> SkillMeta.triggers == []"""
    meta = _fm_and_meta("name: test\ndescription: desc\n")
    assert meta.triggers == []


def test_ac5_3_triggers_not_a_list(capsys):
    """AC-5-3: triggers: "not a list" -> ValueError"""
    with pytest.raises(ValueError, match="triggers"):
        _fm_and_meta('name: test\ndescription: desc\ntriggers: "not a list"\n')
