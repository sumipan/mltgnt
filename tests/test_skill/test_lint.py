"""
tests/test_skill/test_lint.py — lint_skill_meta V1–V12, V13 単体テスト（Issue #1383 U3, #1832, #1828）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.skill.lint import lint_skill_meta


def _path(name: str = "review") -> Path:
    return Path(f"/skills/{name}/SKILL.md")


class TestLintV1V2:
    def test_v1_missing_description(self) -> None:
        errors = lint_skill_meta({"name": "review"}, _path())
        assert any(e.startswith("V1:") for e in errors)

    def test_v2_triggers_not_list(self) -> None:
        errors = lint_skill_meta(
            {"name": "review", "description": "desc", "triggers": "bad"},
            _path(),
        )
        assert any(e.startswith("V2:") for e in errors)


class TestLintV3:
    def test_v3_name_mismatch(self) -> None:
        errors = lint_skill_meta(
            {"name": "foo", "description": "desc"},
            _path("bar"),
        )
        assert any("V3:" in e and "foo" in e and "bar" in e for e in errors)

    def test_v3_name_fallback_matches_dir(self) -> None:
        errors = lint_skill_meta({"description": "desc"}, _path("summarize"))
        assert not any(e.startswith("V3:") for e in errors)


class TestLintV4:
    @pytest.mark.parametrize("bad_value", ["v2", "V1", 1])
    def test_v4_invalid_skill_io(self, bad_value: object) -> None:
        errors = lint_skill_meta(
            {"name": "review", "description": "desc", "skill_io": bad_value},
            _path(),
        )
        assert any(e.startswith("V4:") for e in errors)

    @pytest.mark.parametrize("good_value", ["legacy", "v1"])
    def test_v4_valid_skill_io(self, good_value: str) -> None:
        fm: dict = {"name": "review", "description": "desc", "skill_io": good_value}
        if good_value == "v1":
            fm["produces"] = {"artifacts": [{"path": "out.md"}]}
        errors = lint_skill_meta(fm, _path())
        assert not any(e.startswith("V4:") for e in errors)


class TestLintV5:
    def test_v5_v1_without_produces(self) -> None:
        errors = lint_skill_meta(
            {"name": "review", "description": "desc", "skill_io": "v1"},
            _path(),
        )
        assert any(e.startswith("V5:") for e in errors)


class TestLintV6:
    def test_v6_content_type_not_str(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "skill_io": "v1",
                "produces": {"content_type": 123, "artifacts": [{"path": "out.md"}]},
            },
            _path(),
        )
        assert any(e.startswith("V6:") for e in errors)


class TestLintV7:
    def test_v7_missing_path(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "skill_io": "v1",
                "produces": {"artifacts": [{"role": "primary"}]},
            },
            _path(),
        )
        assert any(e.startswith("V7:") for e in errors)

    def test_v7_path_not_str(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "skill_io": "v1",
                "produces": {"artifacts": [{"path": 42}]},
            },
            _path(),
        )
        assert any(e.startswith("V7:") for e in errors)


class TestLintV8:
    def test_v8_empty_producer(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "consumes": [{"producer": ""}],
            },
            _path(),
        )
        assert any(e.startswith("V8:") for e in errors)

    def test_v8_missing_producer(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "consumes": [{"content_type": "text/plain"}],
            },
            _path(),
        )
        assert any(e.startswith("V8:") for e in errors)


class TestLintV9:
    def test_v9_input_schema_not_dict_for_v1(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "skill_io": "v1",
                "produces": {"artifacts": [{"path": "out.md"}]},
                "input_schema": [],
            },
            _path(),
        )
        assert any(e.startswith("V9:") for e in errors)

    def test_v9_legacy_list_input_schema_allowed(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "input_schema": ["field_a", "field_b"],
            },
            _path(),
        )
        assert not any(e.startswith("V9:") for e in errors)


class TestLintV10V12:
    def test_v10_writes_not_list(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": {"writes": "not-a-list"},
            },
            _path(),
        )
        assert any(
            e == "V10: side_effects.writes must be a list of strings"
            for e in errors
        )

    def test_v10_writes_valid(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": {"writes": ["valid"]},
            },
            _path(),
        )
        assert not any(e.startswith("V10:") for e in errors)

    def test_v11_network_not_list(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": {"network": 123},
            },
            _path(),
        )
        assert any(
            e == "V11: side_effects.network must be a list of strings"
            for e in errors
        )

    def test_v12_mutates_valid(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": {"mutates": ["git", "github"]},
            },
            _path(),
        )
        assert not any(e.startswith("V12:") for e in errors)

    def test_v12_mutates_invalid(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": {"mutates": ["invalid_value"]},
            },
            _path(),
        )
        assert any(
            e
            == "V12: side_effects.mutates values must be from "
            "{config, env, git, github, process}"
            for e in errors
        )

    def test_no_side_effects_key_skips_v10_v12(self) -> None:
        errors = lint_skill_meta(
            {"name": "review", "description": "desc"},
            _path(),
        )
        assert not any(e.startswith(("V10:", "V11:", "V12:")) for e in errors)

    def test_side_effects_not_mapping(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "side_effects": "bad",
            },
            _path(),
        )
        assert any(e == "V10: side_effects must be a mapping" for e in errors)


class TestLintV14:
    def test_v14_scripts_without_side_effects(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "diary-draft"
        skill_dir.mkdir()
        (skill_dir / "scripts").mkdir()
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {"name": "diary-draft", "description": "desc"},
            skill_dir / "SKILL.md",
        )
        assert any(
            e == "V14: skills with scripts/ must declare side_effects" for e in errors
        )

    def test_v14_scripts_with_side_effects(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "diary-draft"
        skill_dir.mkdir()
        (skill_dir / "scripts").mkdir()
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {
                "name": "diary-draft",
                "description": "desc",
                "side_effects": {"writes": ["out.md"]},
            },
            skill_dir / "SKILL.md",
        )
        assert not any(e.startswith("V14:") for e in errors)

    def test_v14_no_scripts_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "simple-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {"name": "simple-skill", "description": "desc"},
            skill_dir / "SKILL.md",
        )
        assert not any(e.startswith("V14:") for e in errors)


class TestLintV13:
    def test_v13_scripts_without_readme(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "diary-draft"
        skill_dir.mkdir()
        (skill_dir / "scripts").mkdir()
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {"name": "diary-draft", "description": "desc"},
            skill_dir / "SKILL.md",
        )
        assert any(e.startswith("V13:") and "(warning)" in e for e in errors)

    def test_v13_scripts_with_readme(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "diary-draft"
        skill_dir.mkdir()
        (skill_dir / "scripts").mkdir()
        (skill_dir / "README.md").write_text("# NAME\n")
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {"name": "diary-draft", "description": "desc"},
            skill_dir / "SKILL.md",
        )
        assert not any(e.startswith("V13:") for e in errors)

    def test_v13_no_scripts_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "simple-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").touch()
        errors = lint_skill_meta(
            {"name": "simple-skill", "description": "desc"},
            skill_dir / "SKILL.md",
        )
        assert not any(e.startswith("V13:") for e in errors)


class TestLintPass:
    def test_legacy_frontmatter_passes(self) -> None:
        errors = lint_skill_meta(
            {"name": "review", "description": "desc", "triggers": ["a"]},
            _path(),
        )
        assert errors == []

    def test_v1_frontmatter_passes(self) -> None:
        errors = lint_skill_meta(
            {
                "name": "review",
                "description": "desc",
                "skill_io": "v1",
                "produces": {
                    "content_type": "text/markdown",
                    "artifacts": [{"path": "report.md", "role": "primary"}],
                    "status_markers": ["ACCEPTED"],
                },
                "consumes": [{"producer": "upstream", "content_type": "text/plain"}],
                "input_schema": {"type": "object"},
            },
            _path(),
        )
        assert errors == []
