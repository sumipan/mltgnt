"""Tests for mltgnt.persona.loader — AC-3 / AC-4 / AC-5 / BC."""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from mltgnt.persona.loader import Persona, load, _parse_sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_persona(body: str, sections: dict[str, str] | None = None) -> Persona:
    """テスト用の Persona インスタンスを直接生成する。"""
    from mltgnt.persona.schema import PersonaFM

    if sections is None:
        sections = _parse_sections(body)

    fm = PersonaFM(name="テスト")
    return Persona(
        name="テスト",
        fm=fm,
        sections=sections,
        body=body,
        path=Path("test.md"),
    )


FULL_BODY = textwrap.dedent("""\
    ## 1. 基本情報

    基本情報の内容。

    ## 2. 価値観

    価値観の内容。

    ## 3. 反応パターン

    反応パターンの内容。

    ## 4. 口調

    口調の内容。

    ## 5. アウトプット形式

    #### critique

    批評フォーマット。

    #### edit

    編集フォーマット。
""")

OUTPUT_FORMAT_BODY = textwrap.dedent("""\
    ## 1. 基本情報

    タチコマ。

    ## 5. アウトプット形式

    #### critique

    批評フォーマットの内容。
    複数行あり。

    #### edit

    編集フォーマットの内容。
""")


# ---------------------------------------------------------------------------
# AC-3: §0 除外
# ---------------------------------------------------------------------------


class TestParseSection0Exclusion:
    def test_3_1_section0_excluded(self):
        body = textwrap.dedent("""\
            ## 0. ファイルの使い方

            この使い方説明は除外される。

            ## 1. 基本情報

            タチコマ。
        """)
        sections = _parse_sections(body)
        assert "ファイルの使い方" not in sections
        assert "基本情報" in sections

    def test_3_2_no_section0_regression(self):
        body = textwrap.dedent("""\
            ## 1. 基本情報

            タチコマ。

            ## 2. 価値観

            好奇心旺盛。
        """)
        sections = _parse_sections(body)
        assert "基本情報" in sections
        assert "価値観" in sections
        assert len(sections) == 2

    def test_3_3_all_persona_files(self):
        from mltgnt.persona import PersonaValidationError

        repo_root = Path(__file__).parent.parent.parent
        persona_dir = repo_root / "personas"
        if not persona_dir.exists():
            pytest.skip("personas/ ディレクトリが存在しない")

        files = list(persona_dir.glob("*.md"))
        if not files:
            pytest.skip("personas/ にファイルがない")

        for path in files:
            try:
                p = load(path)
                assert p is not None
            except PersonaValidationError:
                pytest.fail(f"PersonaValidationError for {path.name}")


# ---------------------------------------------------------------------------
# AC-4: H2 ブロック選択
# ---------------------------------------------------------------------------


class TestFormatPromptWeightSelection:
    def test_4_1_weight_heavy(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("指示", weight="heavy")
        assert "基本情報の内容" in result
        assert "価値観の内容" in result
        assert "反応パターンの内容" in result
        assert "口調の内容" in result
        assert "批評フォーマット" not in result

    def test_4_2_weight_light(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("指示", weight="light")
        assert "指示" in result
        assert "基本情報の内容" not in result

    def test_4_3_default_is_heavy(self):
        persona = _make_persona(FULL_BODY)
        result_default = persona.format_prompt("指示")
        result_heavy = persona.format_prompt("指示", weight="heavy")
        assert result_default == result_heavy

    def test_4_4_unknown_section_warns_and_fallbacks(self):
        body_with_unknown = textwrap.dedent("""\
            ## 1. 未知のセクション

            未知の内容。

            ## 2. 基本情報

            基本情報の内容。
        """)
        persona = _make_persona(body_with_unknown)
        with patch("mltgnt.persona.loader.logger") as mock_logger:
            result = persona.format_prompt("指示", weight="heavy")
            mock_logger.warning.assert_called()
        assert "未知の内容" in result
        assert "基本情報の内容" in result


# ---------------------------------------------------------------------------
# AC-5: 出力形式抽出
# ---------------------------------------------------------------------------


class TestExtractOutputFormat:
    def test_5_1_extract_critique(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "批評フォーマットの内容" in result
        assert "編集フォーマットの内容" not in result

    def test_5_2_default_op_mode(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result_none = persona.extract_output_format(None)
        result_critique = persona.extract_output_format("critique")
        assert result_none == result_critique

    def test_5_3_nonexistent_op_mode_returns_none(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("nonexistent")
        assert result is None

    def test_5_4_critique_does_not_include_edit(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "編集フォーマットの内容" not in result

    def test_5_5_no_output_format_section(self):
        body = textwrap.dedent("""\
            ## 1. 基本情報

            タチコマ。
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is None


# ---------------------------------------------------------------------------
# 後方互換性
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_bc_1_format_prompt_no_weight(self):
        persona = _make_persona(FULL_BODY)
        result = persona.format_prompt("テスト指示")
        assert "テスト指示" in result

    def test_bc_2_build_review_prompt_with_output(self):
        persona = _make_persona(OUTPUT_FORMAT_BODY)
        result = persona.build_review_prompt("critique")
        assert "タチコマ" in result
        assert "批評フォーマットの内容" in result

    def test_bc_2_build_review_prompt_no_output_section(self):
        body = "## 1. 基本情報\n\nタチコマ。"
        persona = _make_persona(body)
        result = persona.build_review_prompt()
        assert "タチコマ" in result


# ---------------------------------------------------------------------------
# AC-1: v2 形式の _parse_sections H3 展開
# ---------------------------------------------------------------------------


import textwrap as _textwrap
from unittest.mock import patch as _patch


class TestParseSectionsV2:
    def test_1_1_v2_expands_h3_heavy_and_reference(self):
        """AC 1-1: v2 形式の H3 展開。重量・参照の H3 がフラット dict に展開される。"""
        body = _textwrap.dedent("""\
            ## 軽量

            サマリ

            ## 重量

            ### 基本情報

            内容A

            ### 価値観

            内容B

            ## 参照

            ### アウトプット形式

            #### critique

            テンプレ

            ### メモ・更新履歴

            更新内容
        """)
        sections = _parse_sections(body)
        assert sections.get("軽量") == "サマリ"
        assert sections.get("基本情報") == "内容A"
        assert sections.get("価値観") == "内容B"
        assert "重量" not in sections
        assert "参照" not in sections
        assert "アウトプット形式" in sections
        assert "#### critique" in sections["アウトプット形式"]
        assert sections.get("メモ・更新履歴") == "更新内容"

    def test_1_2_format_prompt_v2_no_unknown_warning(self):
        """AC 1-2: v2 形式（重量のみ）で format_prompt が WEIGHT_MAP 未定義警告を出さない。"""
        body = _textwrap.dedent("""\
            ## 重量

            ### 基本情報

            基本情報の内容。

            ### 価値観

            価値観の内容。

            ### 反応パターン

            反応パターンの内容。

            ### 口調

            口調の内容。
        """)
        persona = _make_persona(body)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            persona.format_prompt("指示", weight="heavy")
            for call_args in mock_logger.warning.call_args_list:
                args = call_args[0]
                msg = args[0] if args else ""
                assert "WEIGHT_MAP に未定義" not in str(args), \
                    f"WEIGHT_MAP 未定義警告が出てはいけない: {args}"

    def test_1_3_reference_block_h3_expansion(self):
        """AC 1-3: 参照ブロックの H3 展開。"""
        body = _textwrap.dedent("""\
            ## 参照

            ### アウトプット形式

            #### critique

            テンプレ

            ### メモ・更新履歴

            更新内容
        """)
        sections = _parse_sections(body)
        assert "アウトプット形式" in sections
        assert "#### critique" in sections["アウトプット形式"]
        assert "テンプレ" in sections["アウトプット形式"]
        assert sections.get("メモ・更新履歴") == "更新内容"
        assert "参照" not in sections

    def test_1_4_v1_backward_compat(self):
        """AC 1-4: v1 後方互換。H2 直接指定のセクションは従来通り。"""
        body = _textwrap.dedent("""\
            ## 基本情報

            内容

            ## 価値観

            内容B
        """)
        sections = _parse_sections(body)
        assert sections == {"基本情報": "内容", "価値観": "内容B"}

    def test_1_5_v1_numbered_heading(self):
        """AC 1-5: v1 番号付き見出しの正規化は維持される。"""
        body = "## 1. 基本情報\n内容"
        sections = _parse_sections(body)
        assert "基本情報" in sections
        assert sections["基本情報"] == "内容"

    def test_1_6_section0_exclusion_maintained(self):
        """AC 1-6: §0 除外の維持。"""
        body = _textwrap.dedent("""\
            ## 0. ファイルの使い方

            内容

            ## 基本情報

            内容B
        """)
        sections = _parse_sections(body)
        assert "ファイルの使い方" not in sections
        assert sections.get("基本情報") == "内容B"

    def test_1_7_v1_full_section_names_no_warning(self):
        """AC 1-7: v1 形式の完全セクション名（価値観・優先順位 等）でも WEIGHT_MAP 未定義警告を出さない。"""
        body = _textwrap.dedent("""\
            ## 基本情報

            基本情報の内容。

            ## 価値観・優先順位

            価値観の内容。

            ## 反応パターン

            反応パターンの内容。

            ## 口調・語り方

            口調の内容。

            ## 軽量

            軽量テキスト。
        """)
        persona = _make_persona(body)
        with _patch("mltgnt.persona.loader.logger") as mock_logger:
            persona.format_prompt("指示", weight="heavy")
            for call_args in mock_logger.warning.call_args_list:
                args = call_args[0]
                assert "WEIGHT_MAP に未定義" not in str(args), \
                    f"WEIGHT_MAP 未定義警告が出てはいけない: {args}"

    def test_1_8_v1_full_section_names_heavy_excludes_light(self):
        """AC 1-8: v1 形式で weight="heavy" のとき軽量セクションが含まれない。"""
        body = _textwrap.dedent("""\
            ## 基本情報

            基本情報の内容。

            ## 価値観・優先順位

            価値観の内容。

            ## 軽量

            軽量テキスト。
        """)
        persona = _make_persona(body)
        result = persona.format_prompt("指示", weight="heavy")
        assert "基本情報の内容" in result
        assert "軽量テキスト" not in result

    def test_1_9_v1_full_section_names_light_weight(self):
        """AC 1-9: v1 形式で weight="light" のとき軽量セクションのみが含まれる。"""
        body = _textwrap.dedent("""\
            ## 基本情報

            基本情報の内容。

            ## 軽量

            軽量テキスト。
        """)
        persona = _make_persona(body)
        result = persona.format_prompt("指示", weight="light")
        assert "軽量テキスト" in result
        assert "基本情報の内容" not in result


# ---------------------------------------------------------------------------
# AC-2: v2 形式の extract_output_format
# ---------------------------------------------------------------------------


class TestExtractOutputFormatV2:
    def test_2_1_v2_extract_critique(self):
        """AC 2-1: v2 形式（参照 > アウトプット形式）から critique を取得できる。"""
        body = _textwrap.dedent("""\
            ## 参照

            ### アウトプット形式

            #### critique

            【所見】

            #### edit

            【修正案】
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "【所見】" in result
        assert "【修正案】" not in result

    def test_2_2_v2_unknown_mode_returns_none(self):
        """AC 2-2: 存在しないモードは None を返す（silent skip）。"""
        body = _textwrap.dedent("""\
            ## 参照

            ### アウトプット形式

            #### critique

            【所見】
        """)
        persona = _make_persona(body)
        assert persona.extract_output_format("debate") is None

    def test_2_3_v1_extract_output_format(self):
        """AC 2-3: v1 形式（H2 直接アウトプット形式）から取得できる。"""
        body = _textwrap.dedent("""\
            ## アウトプット形式

            #### critique

            内容
        """)
        persona = _make_persona(body)
        result = persona.extract_output_format("critique")
        assert result is not None
        assert "内容" in result


# ---------------------------------------------------------------------------
# AC-3: v2 形式の extract_triage_section
# ---------------------------------------------------------------------------


from mltgnt.persona.triage import extract_triage_section as _extract_triage_section


class TestExtractTriageSectionV2:
    def test_3_1_v2_returns_light_section(self):
        """AC 3-1: v2 形式（## 軽量）のトリアージセクションを返す。"""
        md = _textwrap.dedent("""\
            ## 軽量

            論理的で率直な

            ## 重量

            ### 基本情報

            内容
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "論理的で率直な" in result

    def test_3_2_v1_fallback(self):
        """AC 3-2: v1 形式（## トリアージ用）のフォールバック。"""
        md = _textwrap.dedent("""\
            ## トリアージ用

            トリアージ内容

            ## 基本情報

            内容
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "トリアージ内容" in result

    def test_3_3_neither_returns_none(self):
        """AC 3-3: 両方なし → None を返す。"""
        md = "## 基本情報\n内容のみ"
        assert _extract_triage_section(md) is None

    def test_3_4_both_present_v2_wins(self):
        """AC 3-4: 両方存在する場合、## 軽量 が優先される。"""
        md = _textwrap.dedent("""\
            ## 軽量

            v2内容

            ## トリアージ用

            v1内容
        """)
        result = _extract_triage_section(md)
        assert result is not None
        assert "v2内容" in result
        assert "v1内容" not in result
