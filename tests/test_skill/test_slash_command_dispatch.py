"""
tests/test_skill/test_slash_command_dispatch.py — #208 切り分けテスト。

/persona-create がスキル実行パスに乗らない問題を再現・切り分けする。
仮説:
  ① resolve_skill() がそもそもマッチしない（名前食い違い）
  ② マッチはするが実行パスに乗らない（ルーティング側の問題）
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from mltgnt.skill.loader import discover, load
from mltgnt.skill.matcher import match
from mltgnt.skill.models import SkillMeta
from mltgnt.routing import resolve_skill


# --------------- matcher 単体テスト（仮説①） ---------------

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
    """ハイフン入りスキル名の match() テスト。"""

    async def test_persona_create_matches(self) -> None:
        """① /persona-create がそもそもマッチするか"""
        result = await match("/persona-create 古賀史健", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "persona-create"
        assert args == "古賀史健"

    async def test_persona_create_no_args(self) -> None:
        """/persona-create 引数なしでもマッチ"""
        result = await match("/persona-create", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "persona-create"
        assert args == ""

    async def test_triple_hyphen_name(self) -> None:
        """3段ハイフン名でもマッチ"""
        result = await match("/diary-review-sakuma", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is not None
        assert result[0].name == "diary-review-sakuma"

    async def test_persona_create_filtered_by_persona_skills(self) -> None:
        """persona_skills に含まれない場合は None"""
        result = await match(
            "/persona-create 古賀史健",
            SKILLS_WITH_HYPHEN,
            persona_skills=["review"],
        )
        assert result is None

    async def test_persona_create_allowed_by_persona_skills(self) -> None:
        """persona_skills に含まれればマッチ"""
        result = await match(
            "/persona-create 古賀史健",
            SKILLS_WITH_HYPHEN,
            persona_skills=["persona-create"],
        )
        assert result is not None
        assert result[0].name == "persona-create"


# --------------- Slack 経由入力の前処理テスト ---------------

class TestSlackInputEdgeCases:
    """Slack メッセージ特有の入力パターンでマッチが落ちないか。
    LLM フォールバックをモックし、スラッシュパターン単体の挙動を検証する。
    """

    async def test_leading_whitespace(self) -> None:
        """先頭にスペースがあるとスラッシュパターンにマッチしない（正常動作の確認）"""
        with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
            result = await match(" /persona-create 古賀史健", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is None, "先頭空白があると /name パターンにマッチしないのは仕様通り"

    async def test_leading_newline(self) -> None:
        """先頭に改行があるとスラッシュパターンにマッチしない"""
        with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
            result = await match("\n/persona-create 古賀史健", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is None, "先頭改行は仕様通り None"

    async def test_multiline_with_slash_on_first_line(self) -> None:
        """/name が先頭行にあり、後続に改行テキストがある場合"""
        result = await match("/persona-create 古賀史健\nよろしくお願いします", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "persona-create"
        assert "古賀史健" in args
        assert "よろしくお願いします" in args

    async def test_slash_in_middle_of_text(self) -> None:
        """テキスト中に /name があってもスラッシュパターンにマッチしない（先頭のみ）"""
        with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
            result = await match("今日 /persona-create を使いたい", SKILLS_WITH_HYPHEN, persona_skills=None)
        assert result is None


# --------------- discover + match 統合テスト（仮説① ファイルシステム） ---------------

PERSONA_CREATE_SKILL_MD = """\
---
name: persona-create
description: >
  人物のペルソナファイルを自動生成する。
argument_hint: "<人物名>"
model: null
---

本文ここ
"""


def _write_skill(tmp_path: Path, rel: str, content: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestDiscoverAndMatchIntegration:
    """discover() で見つけた skills dict を match() に渡す統合テスト。"""

    def test_discover_finds_hyphenated_skill(self, tmp_path: Path) -> None:
        """persona-create ディレクトリの SKILL.md が discover される"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        assert "persona-create" in skills

    async def test_discover_then_match(self, tmp_path: Path) -> None:
        """discover → match のパイプラインで /persona-create がマッチ"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        result = await match("/persona-create 古賀史健", skills, persona_skills=None)
        assert result is not None
        meta, args = result
        assert meta.name == "persona-create"
        assert args == "古賀史健"

    async def test_discover_then_match_then_load(self, tmp_path: Path) -> None:
        """discover → match → load フルパイプライン"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        skills = discover([tmp_path])
        result = await match("/persona-create 古賀史健", skills, persona_skills=None)
        assert result is not None
        meta, args = result
        skill_file = load(meta)
        assert skill_file.meta.name == "persona-create"
        assert "本文ここ" in skill_file.body


# --------------- resolve_skill 統合テスト（仮説② ルーティング側） ---------------

class TestResolveSkillIntegration:
    """resolve_skill() の結合テスト。ファイルシステムから実際に解決する。"""

    async def test_resolve_persona_create(self, tmp_path: Path) -> None:
        """resolve_skill が /persona-create を解決できる"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill("/persona-create 古賀史健", [tmp_path])
        assert result is not None
        skill_file, args = result
        assert skill_file.meta.name == "persona-create"
        assert args == "古賀史健"

    async def test_resolve_persona_create_with_persona_filter_pass(self, tmp_path: Path) -> None:
        """persona_skills に含まれていれば解決"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill(
            "/persona-create 古賀史健",
            [tmp_path],
            persona_skills=["persona-create"],
        )
        assert result is not None

    async def test_resolve_persona_create_with_persona_filter_block(self, tmp_path: Path) -> None:
        """persona_skills に含まれてなければ None"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        result = await resolve_skill(
            "/persona-create 古賀史健",
            [tmp_path],
            persona_skills=["review"],
        )
        assert result is None

    async def test_resolve_plain_text_returns_none(self, tmp_path: Path) -> None:
        """スラッシュなし平文は LLM をモックした場合 None（スラッシュ/triggers マッチなし）"""
        _write_skill(tmp_path, "persona-create/SKILL.md", PERSONA_CREATE_SKILL_MD)
        with patch("mltgnt.skill.matcher._match_by_llm", new=AsyncMock(return_value=None)):
            result = await resolve_skill("ペルソナ作って", [tmp_path])
        assert result is None

    async def test_resolve_with_empty_paths(self) -> None:
        """空パスリスト → None"""
        result = await resolve_skill("/persona-create foo", [])
        assert result is None

    async def test_resolve_with_real_skills_dir(self) -> None:
        """実際の skills/ ディレクトリから解決（SKILL.md が存在する場合のみ）"""
        real_skills_dir = Path("/Users/ngystks/Github/diary/skills")
        skill_md = real_skills_dir / "persona-create" / "SKILL.md"
        if not skill_md.exists():
            pytest.skip("skills/persona-create/SKILL.md が存在しない")
        result = await resolve_skill("/persona-create テスト人物", [real_skills_dir])
        assert result is not None, (
            "実際の skills/persona-create/SKILL.md が存在するのに "
            "resolve_skill が None を返した — discover or match に問題あり"
        )
        skill_file, args = result
        assert skill_file.meta.name == "persona-create"
        assert args == "テスト人物"
