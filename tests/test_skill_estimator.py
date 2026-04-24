"""tests/test_skill_estimator.py — estimate_skill の受け入れ条件テスト。"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.skill import estimate_skill


CALENDAR_SKILL_MD = """\
---
name: calendar
description: カレンダー操作スキル
argument_hint: ""
triggers:
  - スケジュール
  - カレンダー
---
カレンダー操作を行います。
"""

PERSONA_SKILL_MD = """\
---
name: persona-create
description: 人物像作成スキル
argument_hint: ""
triggers:
  - 人物像作成
---
人物像を作成します。
"""


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """calendar と persona-create の SKILL.md を含む一時ディレクトリを返す。"""
    cal_dir = tmp_path / "calendar"
    cal_dir.mkdir()
    (cal_dir / "SKILL.md").write_text(CALENDAR_SKILL_MD, encoding="utf-8")

    persona_dir = tmp_path / "persona-create"
    persona_dir.mkdir()
    (persona_dir / "SKILL.md").write_text(PERSONA_SKILL_MD, encoding="utf-8")

    return tmp_path


# ── 正常系 ──────────────────────────────────────────────────────────────────

def test_match_schedule_keyword(skill_dir: Path) -> None:
    """AC-1: 「スケジュール」で calendar にマッチする。"""
    result = estimate_skill("明日のスケジュール教えて", [skill_dir])
    assert result == "calendar"


def test_match_calendar_keyword(skill_dir: Path) -> None:
    """AC-2: 「カレンダー」で calendar にマッチする。"""
    result = estimate_skill("カレンダー確認して", [skill_dir])
    assert result == "calendar"


def test_match_persona_create(skill_dir: Path) -> None:
    """AC-3: 「人物像作成」で persona-create にマッチする。"""
    result = estimate_skill("人物像作成 古賀史健", [skill_dir])
    assert result == "persona-create"


# ── 異常系・境界値 ──────────────────────────────────────────────────────────

def test_no_match_returns_none(skill_dir: Path) -> None:
    """AC-4: どの triggers にもマッチしない場合 None を返す。"""
    result = estimate_skill("今日は天気がいいね", [skill_dir])
    assert result is None


def test_empty_instruction_returns_none(skill_dir: Path) -> None:
    """AC-5: 空文字列で None を返す。"""
    result = estimate_skill("", [skill_dir])
    assert result is None


def test_empty_skill_paths_returns_none() -> None:
    """AC-6: 空のパスリストで None を返す。"""
    result = estimate_skill("予定", [])
    assert result is None


def test_generic_yotei_no_match(skill_dir: Path) -> None:
    """AC-7: 汎用語「予定」は triggers に含まれないため None を返す。"""
    result = estimate_skill("26日14時で予定を入れて", [skill_dir])
    assert result is None


# ── 既存機能への影響 ─────────────────────────────────────────────────────────

def test_import_from_mltgnt_skill() -> None:
    """AC-9: `from mltgnt.skill import estimate_skill` で import できること。"""
    from mltgnt.skill import estimate_skill as fn  # noqa: PLC0415
    assert callable(fn)
