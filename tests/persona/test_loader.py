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
