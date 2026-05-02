"""Tests for mltgnt.persona.compress — LLM 圧縮・ハッシュ・再生成・drift 検出。"""
from __future__ import annotations

import hashlib
import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_llm_result(ok: bool = True, stdout: str = "圧縮テキスト", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.ok = ok
    r.stdout = stdout
    r.stderr = stderr
    return r


# v2.1 形式のモックレスポンス（_validate_v21_light_block を通過できる）
_V21_MOCK_RESPONSE = textwrap.dedent("""\
    フチコマは好奇心旺盛で論理的な人物。新しい技術に対して積極的に向き合う。

    **口調** — 簡潔でテンポよく話し、専門用語を適切に使う。
    **価値観** — 効率と正確さを重視し、曖昧さを嫌う。
    **好意的反応** — 論理的な提案や新技術の話題には前のめりになる。
    **引っかかる** — 根拠のない主張や非効率な手順には不満を示す。
""")

# 発言例付きの v2.1 形式
_V21_MOCK_WITH_SPEECH = textwrap.dedent("""\
    フチコマは好奇心旺盛で論理的な人物。新しい技術に対して積極的に向き合う。

    **口調** — 簡潔でテンポよく話し、専門用語を適切に使う。
    **価値観** — 効率と正確さを重視し、曖昧さを嫌う。
    **好意的反応** — 論理的な提案や新技術の話題には前のめりになる。
    **引っかかる** — 根拠のない主張や非効率な手順には不満を示す。
    **発言例**
    > それ、もう少し根拠を整理してから話してほしいな。
""")

V2_PERSONA = textwrap.dedent("""\
    ---
    persona:
      name: テスト太郎
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 軽量

    既存の軽量テキスト。

    ## 重量

    詳細な人物像のテキスト。価値観・反応パターン・口調など。
    このペルソナは非常に好奇心旺盛で、新しいことに挑戦するのが好きです。

    ## 参照

    参照リンクや補足情報。
""")

V2_PERSONA_EMPTY_LIGHT = textwrap.dedent("""\
    ---
    persona:
      name: テスト太郎
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 軽量

    ## 重量

    詳細な人物像のテキスト。価値観・反応パターン・口調など。

    ## 参照

    参照リンクや補足情報。
""")

V1_PERSONA = textwrap.dedent("""\
    ---
    persona:
      name: テスト太郎
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## 基本情報

    v1 形式のペルソナ。重量ブロックなし。

    ## 価値観

    テスト好き。
""")


# ---------------------------------------------------------------------------
# AC-2: compute_block_hash
# ---------------------------------------------------------------------------


class TestComputeBlockHash:
    def test_returns_64char_hex(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        result = compute_block_hash("テスト文字列")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_strip_normalization(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        assert compute_block_hash("テスト文字列\n") == compute_block_hash("テスト文字列")

    def test_crlf_normalization(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        assert compute_block_hash("行1\r\n行2") == compute_block_hash("行1\n行2")

    def test_empty_string(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        result = compute_block_hash("")
        expected = hashlib.sha256("".encode("utf-8")).hexdigest()
        assert result == expected


# ---------------------------------------------------------------------------
# AC-1: compress_heavy_to_light
# ---------------------------------------------------------------------------


class TestCompressHeavyToLight:
    def test_normal_long_text(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        heavy = "あ" * 1500
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = compress_heavy_to_light(heavy)
        assert isinstance(result, str)
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_normal_short_text(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        heavy = "短い重量ブロック" * 5
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            result = compress_heavy_to_light(heavy)
        mock_call.assert_called_once()
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_empty_input_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with pytest.raises(RuntimeError):
            compress_heavy_to_light("")

    def test_llm_failure_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("ghdag.llm.call", side_effect=TimeoutError("timeout")):
            with pytest.raises(RuntimeError, match="timeout"):
                compress_heavy_to_light("重量テキスト")

    def test_llm_ok_false_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("ghdag.llm.call", return_value=_make_llm_result(ok=False, stderr="engine error")):
            with pytest.raises(RuntimeError, match="engine error"):
                compress_heavy_to_light("重量テキスト")

    def test_engine_and_model_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("テスト", engine="claude", model="claude-haiku-4-5")
        _, kwargs = mock_call.call_args
        assert kwargs.get("engine") == "claude"
        assert kwargs.get("model") == "claude-haiku-4-5"

    def test_timeout_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("テスト", timeout=60)
        _, kwargs = mock_call.call_args
        assert kwargs.get("timeout") == 60


# ---------------------------------------------------------------------------
# AC-3: regenerate_light_block
# ---------------------------------------------------------------------------


class TestRegenerateLightBlock:
    def test_first_generation(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.is_first_generation is True
        assert result.old_hash == ""
        assert result.light_text == _V21_MOCK_RESPONSE.strip()
        content = persona_file.read_text(encoding="utf-8")
        assert "**口調**" in content

    def test_regeneration_changed(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.changed is True
        assert result.old_hash != result.new_hash
        content = persona_file.read_text(encoding="utf-8")
        assert "**口調**" in content

    def test_regeneration_unchanged(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        # まず v2.1 形式の軽量ブロックを持つペルソナを準備
        v2_persona_v21_light = """---
persona:
  name: テスト太郎
ops:
  engine: claude
  model: claude-sonnet-4-6
---

## 軽量

フチコマは好奇心旺盛で論理的な人物。新しい技術に対して積極的に向き合う。

**口調** — 簡潔でテンポよく話し、専門用語を適切に使う。
**価値観** — 効率と正確さを重視し、曖昧さを嫌う。
**好意的反応** — 論理的な提案や新技術の話題には前のめりになる。
**引っかかる** — 根拠のない主張や非効率な手順には不満を示す。

## 重量

詳細な人物像のテキスト。価値観・反応パターン・口調など。
このペルソナは非常に好奇心旺盛で、新しいことに挑戦するのが好きです。

## 参照

参照リンクや補足情報。
"""
        persona_file.write_text(v2_persona_v21_light, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.changed is False
        assert result.old_hash == result.new_hash

    def test_drift_warning_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            with caplog.at_level(logging.WARNING, logger="mltgnt.persona.compress"):
                result = regenerate_light_block(persona_file)
        if result.changed:
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) > 0
            log_text = " ".join(r.message for r in warning_records)
            assert result.old_hash[:8] in log_text or result.new_hash[:8] in log_text

    def test_file_integrity(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        content = persona_file.read_text(encoding="utf-8")
        assert "persona:" in content
        assert "name: テスト太郎" in content
        assert "詳細な人物像のテキスト" in content
        assert "## 重量" in content
        assert "参照リンクや補足情報" in content
        assert "## 参照" in content

    def test_invalid_v2_raises_value_error(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V1_PERSONA, encoding="utf-8")
        with pytest.raises(ValueError, match="v2 形式ではありません"):
            regenerate_light_block(persona_file)

    def test_result_persona_name(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.persona_name == "テスト太郎"


# ---------------------------------------------------------------------------
# AC-4: 統合テスト
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_file_has_light_block_under_1500_chars(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block, _split_h2_blocks
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        content = persona_file.read_text(encoding="utf-8")
        from mltgnt.persona.frontmatter import split_yaml_frontmatter
        _, body = split_yaml_frontmatter(content)
        blocks = _split_h2_blocks(body)
        light_text = blocks.get("軽量", "")
        assert len(light_text) <= 1500

    def test_loader_compatible_after_regeneration(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        from mltgnt.persona.loader import load
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        persona = load(persona_file)
        assert persona.name == "テスト太郎"


# ---------------------------------------------------------------------------
# 新規: LIGHT_BLOCK_MAX_CHARS 定数テスト (AC-2 #1)
# ---------------------------------------------------------------------------


class TestLightBlockMaxChars:
    def test_constant_is_1500(self) -> None:
        from mltgnt.persona.compress import LIGHT_BLOCK_MAX_CHARS
        assert LIGHT_BLOCK_MAX_CHARS == 1500


# ---------------------------------------------------------------------------
# 新規: _validate_v21_light_block テスト (AC-2 #5-7)
# ---------------------------------------------------------------------------


class TestValidateV21LightBlock:
    """_validate_v21_light_block の正常・異常系テスト。"""

    def test_valid_standard_block(self) -> None:
        """正常: 標準的な v2.1 ブロックはエラーなし。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        _validate_v21_light_block(_V21_MOCK_RESPONSE)  # エラーなし

    def test_valid_with_speech_examples(self) -> None:
        """正常: 発言例付きの v2.1 ブロックはエラーなし。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        _validate_v21_light_block(_V21_MOCK_WITH_SPEECH)  # エラーなし

    def test_error_no_lead_text(self) -> None:
        """異常: リード文なし（**口調** から始まる）→ ValueError('リード文')。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        no_lead = """**口調** — 簡潔でテンポよく話す。
**価値観** — 効率を重視する。
**好意的反応** — 論理的な提案に喜ぶ。
**引っかかる** — 根拠のない主張に不満。
"""
        with pytest.raises(ValueError, match="リード文"):
            _validate_v21_light_block(no_lead)

    def test_error_missing_section_chotyp(self) -> None:
        """異常: **口調** がない → ValueError に '口調' が含まれる。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = """フチコマは好奇心旺盛な人物。

**価値観** — 効率を重視する。
**好意的反応** — 論理的な提案に喜ぶ。
**引っかかる** — 根拠のない主張に不満。
"""
        with pytest.raises(ValueError, match="口調"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_kachikan(self) -> None:
        """異常: **価値観** がない → ValueError に '価値観' が含まれる。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = """フチコマは好奇心旺盛な人物。

**口調** — 簡潔に話す。
**好意的反応** — 論理的な提案に喜ぶ。
**引っかかる** — 根拠のない主張に不満。
"""
        with pytest.raises(ValueError, match="価値観"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_koitekireaction(self) -> None:
        """異常: **好意的反応** がない → ValueError に '好意的反応' が含まれる。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = """フチコマは好奇心旺盛な人物。

**口調** — 簡潔に話す。
**価値観** — 効率を重視する。
**引っかかる** — 根拠のない主張に不満。
"""
        with pytest.raises(ValueError, match="好意的反応"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_hikkakaru(self) -> None:
        """異常: **引っかかる** がない → ValueError に '引っかかる' が含まれる。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = """フチコマは好奇心旺盛な人物。

**口調** — 簡潔に話す。
**価値観** — 効率を重視する。
**好意的反応** — 論理的な提案に喜ぶ。
"""
        with pytest.raises(ValueError, match="引っかかる"):
            _validate_v21_light_block(missing_section)

    def test_error_speech_example_without_quote(self) -> None:
        """異常: **発言例** の後に > 行がない → ValueError。"""
        from mltgnt.persona.compress import _validate_v21_light_block
        bad_speech = """フチコマは好奇心旺盛な人物。

**口調** — 簡潔に話す。
**価値観** — 効率を重視する。
**好意的反応** — 論理的な提案に喜ぶ。
**引っかかる** — 根拠のない主張に不満。
**発言例**
それ、もう少し根拠を整理してほしいな。
"""
        with pytest.raises(ValueError):
            _validate_v21_light_block(bad_speech)


# ---------------------------------------------------------------------------
# 新規: regenerate_light_block が v2.1 バリデーションを呼ぶことの確認
# ---------------------------------------------------------------------------


class TestRegenerateLightBlockV21Validation:
    def test_invalid_v21_raises_value_error(self, tmp_path: Path) -> None:
        """LLM がバリデーション不通過な応答を返した場合 ValueError になる。"""
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        bad_response = "**口調** — リード文なし。\n**価値観** — 効率。\n**好意的反応** — OK。\n**引っかかる** — NG。"
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=bad_response)):
            with pytest.raises(ValueError, match="リード文"):
                regenerate_light_block(persona_file)
