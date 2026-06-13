"""
tests/test_skill/test_tool_validation.py — Issue #2090: tools フロントマターと Tool バリデーション。
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.bridges.files_adapter import md_read
from mltgnt.skill.loader import _build_meta, validate_tool_refs
from mltgnt.skill.models import SkillLoadError, SkillMeta


def _fm_and_meta(yaml_content: str, tmp_path: Path) -> SkillMeta:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(f"---\n{yaml_content}\n---\nbody\n", encoding="utf-8")
    md = md_read(skill_path.name, repo_root=tmp_path)
    return _build_meta(md.frontmatter, skill_path)


def _meta(name: str, tools: list[str], path: Path | None = None) -> SkillMeta:
    return SkillMeta(
        name=name,
        description="desc",
        argument_hint="",
        model=None,
        path=path or Path(f"/fake/skills/{name}/SKILL.md"),
        tools=tools,
    )


class TestToolsFrontmatterParse:
    def test_tools_list(self, tmp_path: Path) -> None:
        """tools: [existing_tool] -> SkillMeta.tools == ['existing_tool']"""
        meta = _fm_and_meta(
            'name: test\ndescription: desc\ntools:\n  - existing_tool\n',
            tmp_path,
        )
        assert meta.tools == ["existing_tool"]

    def test_tools_missing(self, tmp_path: Path) -> None:
        """tools キー未指定 -> SkillMeta.tools == []"""
        meta = _fm_and_meta("name: test\ndescription: desc\n", tmp_path)
        assert meta.tools == []

    def test_tools_not_a_list(self, tmp_path: Path) -> None:
        """tools: "not_a_list" -> ValueError"""
        with pytest.raises(ValueError, match="tools"):
            _fm_and_meta('name: test\ndescription: desc\ntools: "not_a_list"\n', tmp_path)


class TestValidateToolRefs:
    def test_unknown_tool_raises(self, tmp_path: Path) -> None:
        """未知 Tool 参照で SkillLoadError（スキル名と Tool 名を含む）"""
        skills = {"bad-skill": _meta("bad-skill", ["nonexistent_tool"])}
        mock_result = MagicMock(returncode=0, stdout=json.dumps({"tools": []}), stderr="")
        with patch("mltgnt.skill.loader.subprocess.run", return_value=mock_result):
            with pytest.raises(SkillLoadError) as exc_info:
                validate_tool_refs(skills, tmp_path)
        msg = str(exc_info.value)
        assert "bad-skill" in msg
        assert "nonexistent_tool" in msg

    def test_empty_tools_skipped(self, tmp_path: Path) -> None:
        """tools が空のスキルはバリデーションをスキップ"""
        skills = {"no-tools": _meta("no-tools", [])}
        with patch("mltgnt.skill.loader.subprocess.run") as mock_run:
            validate_tool_refs(skills, tmp_path)
            mock_run.assert_not_called()

    def test_existing_tool_passes(self, tmp_path: Path) -> None:
        """既知 Tool 参照はエラーなし"""
        skills = {"good-skill": _meta("good-skill", ["existing_tool"])}
        mock_result = MagicMock(
            returncode=0,
            stdout=json.dumps({"tools": [{"name": "existing_tool"}]}),
            stderr="",
        )
        with patch("mltgnt.skill.loader.subprocess.run", return_value=mock_result):
            validate_tool_refs(skills, tmp_path)

    def test_ghdag_failure_raises(self, tmp_path: Path) -> None:
        """ghdag tools list 失敗時に stderr を含む SkillLoadError"""
        skills = {"any": _meta("any", ["tool-a"])}
        mock_result = MagicMock(returncode=1, stdout="", stderr="command failed: not found")
        with patch("mltgnt.skill.loader.subprocess.run", return_value=mock_result):
            with pytest.raises(SkillLoadError) as exc_info:
                validate_tool_refs(skills, tmp_path)
        assert "command failed: not found" in str(exc_info.value)

    def test_multiple_unknown_tools_aggregated(self, tmp_path: Path) -> None:
        """複数スキルの未知 Tool を 1 つの SkillLoadError に集約"""
        skills = {
            "skill-a": _meta("skill-a", ["unknown-a"]),
            "skill-b": _meta("skill-b", ["unknown-b", "unknown-c"]),
        }
        mock_result = MagicMock(returncode=0, stdout=json.dumps({"tools": []}), stderr="")
        with patch("mltgnt.skill.loader.subprocess.run", return_value=mock_result):
            with pytest.raises(SkillLoadError) as exc_info:
                validate_tool_refs(skills, tmp_path)
        msg = str(exc_info.value)
        assert "skill-a" in msg
        assert "unknown-a" in msg
        assert "skill-b" in msg
        assert "unknown-b" in msg
        assert "unknown-c" in msg
