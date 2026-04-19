"""tests/test_persona.py — mltgnt.persona のテスト。"""

from __future__ import annotations


from mltgnt.persona import Persona, validate_persona
from mltgnt.persona.schema import (
    REQUIRED_SECTIONS,
    PersonaFM,
    validate_sections,
)


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_full_body() -> str:
    """REQUIRED_SECTIONS をすべて含む本文を生成。"""
    lines = []
    for sec in REQUIRED_SECTIONS:
        lines.append(f"## {sec}")
        lines.append(f"{sec} の内容")
        lines.append("")
    return "\n".join(lines)


def _make_fm(name: str = "test") -> PersonaFM:
    return PersonaFM(name=name)


# ---------------------------------------------------------------------------
# AC1: validate_sections
# ---------------------------------------------------------------------------

class TestValidateSections:
    """AC1: validate_sections の required_sections 引数テスト。"""

    def test_ac1_1_none_all_sections_present(self):
        """AC1-1: required_sections=None, すべての必須セクションあり → ok=True, warnings=[]"""
        body = _make_full_body()
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=None)
        assert result.ok is True
        assert result.warnings == []

    def test_ac1_2_none_missing_kihon_joho(self):
        """AC1-2: required_sections=None, 基本情報がない → warnings に「基本情報」を含む"""
        # 基本情報以外のセクションを含む本文
        body = "\n".join(
            f"## {sec}\n{sec} の内容\n"
            for sec in REQUIRED_SECTIONS
            if sec != "基本情報"
        )
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=None)
        assert any("基本情報" in w for w in result.warnings)

    def test_ac1_3_empty_tuple_skip(self):
        """AC1-3: required_sections=() → ok=True, warnings=[] (スキップ)"""
        body = ""  # 空でも
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=())
        assert result.ok is True
        assert result.warnings == []

    def test_ac1_4_custom_section_present(self):
        """AC1-4: required_sections=("カスタム",), body に ## カスタム あり → ok=True, warnings=[]"""
        body = "## カスタム\nカスタム内容"
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=("カスタム",))
        assert result.ok is True
        assert result.warnings == []

    def test_ac1_5_custom_section_missing(self):
        """AC1-5: required_sections=("カスタム",), body に ## カスタム なし → warnings に「カスタム」"""
        body = "## 別のセクション\n内容"
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=("カスタム",))
        assert any("カスタム" in w for w in result.warnings)

    def test_numbered_section_format(self):
        """番号付きセクション形式（## 1. 基本情報）でもマッチする。"""
        body = "\n".join(
            f"## {i+1}. {sec}\n{sec} の内容\n"
            for i, sec in enumerate(REQUIRED_SECTIONS)
        )
        fm = _make_fm()
        result = validate_sections(body, fm, required_sections=None)
        assert result.ok is True
        assert result.warnings == []


# ---------------------------------------------------------------------------
# AC2: validate_persona
# ---------------------------------------------------------------------------

class TestValidatePersona:
    """AC2: validate_persona の required_sections 引数テスト。"""

    def test_ac2_1_none_valid_persona(self):
        """AC2-1: required_sections=None, 有効なペルソナ → messages=[]"""
        body = _make_full_body()
        fm = _make_fm("alice")
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona, required_sections=None)
        assert messages == []

    def test_ac2_2_custom_section_missing(self):
        """AC2-2: required_sections=("存在しないセクション",) → messages に「存在しないセクション」"""
        body = _make_full_body()
        fm = _make_fm("alice")
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona, required_sections=("存在しないセクション",))
        assert any("存在しないセクション" in m for m in messages)

    def test_ac2_3_empty_tuple_no_section_messages(self):
        """AC2-3: required_sections=() → セクション関連メッセージなし"""
        body = ""  # セクションなし
        fm = _make_fm("alice")
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona, required_sections=())
        # セクション関連の警告が含まれないことを確認
        section_msgs = [m for m in messages if "セクション" in m]
        assert section_msgs == []


# ---------------------------------------------------------------------------
# AC3: 後方互換
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """AC3: 後方互換テスト。"""

    def test_ac3_required_sections_constant_exists(self):
        """REQUIRED_SECTIONS 定数が schema.py に存在する。"""
        assert isinstance(REQUIRED_SECTIONS, tuple)
        assert len(REQUIRED_SECTIONS) > 0

    def test_ac3_validate_sections_two_args(self):
        """validate_sections(body, fm) の 2 引数呼び出しが動作する。"""
        body = _make_full_body()
        fm = _make_fm()
        result = validate_sections(body, fm)
        assert isinstance(result.ok, bool)

    def test_ac3_validate_persona_no_required_sections(self):
        """validate_persona(persona) の required_sections 省略呼び出しが動作する。"""
        body = _make_full_body()
        fm = _make_fm("alice")
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona)
        assert isinstance(messages, list)

    def test_ac3_required_sections_content(self):
        """REQUIRED_SECTIONS の内容を確認する。"""
        expected = ("基本情報", "価値観", "反応パターン", "口調", "アウトプット形式")
        assert REQUIRED_SECTIONS == expected


# ---------------------------------------------------------------------------
# 追加: available_skills テスト
# ---------------------------------------------------------------------------

class TestAvailableSkills:
    """validate_persona の available_skills 引数テスト。"""

    def test_unknown_skill_warning(self):
        """利用不可のスキルに警告が出る。"""
        body = _make_full_body()
        fm = PersonaFM(name="alice", skills=["search", "unknown_skill"])
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona, available_skills=["search"])
        assert any("unknown_skill" in m for m in messages)

    def test_known_skills_no_warning(self):
        """すべてのスキルが利用可能なら警告なし。"""
        body = _make_full_body()
        fm = PersonaFM(name="alice", skills=["search"])
        persona = Persona(name="alice", fm=fm, body=body)
        messages = validate_persona(persona, available_skills=["search", "calendar"])
        skill_msgs = [m for m in messages if "スキル" in m]
        assert skill_msgs == []
