"""
tests/test_skill/test_loader.py — loader.discover / loader.load のユニットテスト。

設計: Issue #124 §8 AC-1, AC-2
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.skill.loader import discover, load
from mltgnt.skill.models import SkillMeta


# --- ヘルパー ---

def _write_skill(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


FULL_SKILL_MD = """\
---
name: review
description: 日記ファイルをレビューする
argument_hint: "[target-file]"
model: null
---

本文ここ
"""

NO_NAME_SKILL_MD = """\
---
description: 要約スキル
---

本文
"""

NO_DESCRIPTION_SKILL_MD = """\
---
name: bad
---

本文
"""

INVALID_YAML_MD = """\
---
name: broken
description: [unclosed
---

本文
"""

NO_FRONTMATTER_MD = "本文のみ\n"


# --- AC-1: SkillMeta パース ---

class TestSkillMetaParse:
    def test_full_fields(self, tmp_path: Path) -> None:
        """AC-1-1: 全フィールドが揃った SKILL.md"""
        p = _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        assert "review" in skills
        meta = skills["review"]
        assert meta.name == "review"
        assert meta.description == "日記ファイルをレビューする"
        assert meta.argument_hint == "[target-file]"
        assert meta.model is None
        assert meta.path == p.resolve()

    def test_name_fallback_to_dir(self, tmp_path: Path) -> None:
        """AC-1-2: name 省略時はディレクトリ名 fallback"""
        _write_skill(tmp_path, "summarize/SKILL.md", NO_NAME_SKILL_MD)
        skills = discover([tmp_path])
        assert "summarize" in skills
        assert skills["summarize"].name == "summarize"

    def test_missing_description_skipped(self, tmp_path: Path, capsys) -> None:
        """AC-1-3: description 省略はパースエラーでスキップ"""
        _write_skill(tmp_path, "bad/SKILL.md", NO_DESCRIPTION_SKILL_MD)
        skills = discover([tmp_path])
        assert skills == {}
        captured = capsys.readouterr()
        assert "パースエラー" in captured.err

    def test_invalid_yaml_skipped(self, tmp_path: Path, capsys) -> None:
        """AC-1-4: 不正 YAML はスキップ"""
        _write_skill(tmp_path, "broken/SKILL.md", INVALID_YAML_MD)
        skills = discover([tmp_path])
        assert skills == {}
        captured = capsys.readouterr()
        assert "パースエラー" in captured.err

    def test_no_frontmatter_skipped(self, tmp_path: Path, capsys) -> None:
        """AC-1-5: frontmatter なしはスキップ"""
        _write_skill(tmp_path, "nofront/SKILL.md", NO_FRONTMATTER_MD)
        skills = discover([tmp_path])
        assert skills == {}
        captured = capsys.readouterr()
        assert "パースエラー" in captured.err


# --- AC-2: discover ---

class TestDiscover:
    def test_multiple_skills(self, tmp_path: Path) -> None:
        """AC-2-1: 複数スキルが発見できる"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        _write_skill(tmp_path, "summarize/SKILL.md", NO_NAME_SKILL_MD)
        skills = discover([tmp_path])
        assert set(skills.keys()) == {"review", "summarize"}

    def test_empty_directory(self, tmp_path: Path) -> None:
        """AC-2-2: 空ディレクトリ → {}"""
        skills = discover([tmp_path])
        assert skills == {}

    def test_nonexistent_path(self, tmp_path: Path, capsys) -> None:
        """AC-2-3: 存在しないパス → {} + 警告"""
        skills = discover([tmp_path / "nonexistent"])
        assert skills == {}
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_duplicate_name_first_wins(self, tmp_path: Path, capsys) -> None:
        """AC-2-4: 同名スキルが複数パスに存在 → 先勝ち"""
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _write_skill(dir_a, "review/SKILL.md", FULL_SKILL_MD)
        _write_skill(dir_b, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([dir_a, dir_b])
        assert len(skills) == 1
        assert skills["review"].path.parent.parent == dir_a.resolve()
        captured = capsys.readouterr()
        assert "重複" in captured.err

    def test_ignores_non_skill_files(self, tmp_path: Path) -> None:
        """AC-2-5: SKILL.md 以外は無視"""
        _write_skill(tmp_path, "review/skill.yaml", "name: review\n")
        skills = discover([tmp_path])
        assert skills == {}

    def test_load_returns_skill_file(self, tmp_path: Path) -> None:
        """load() が SkillFile を返す"""
        _write_skill(tmp_path, "review/SKILL.md", FULL_SKILL_MD)
        skills = discover([tmp_path])
        sf = load(skills["review"])
        assert sf.meta.name == "review"
        assert "本文ここ" in sf.body

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """load() が FileNotFoundError を投げる"""
        meta = SkillMeta(
            name="ghost",
            description="ghost",
            argument_hint="",
            model=None,
            path=tmp_path / "ghost" / "SKILL.md",
        )
        with pytest.raises(FileNotFoundError):
            load(meta)
