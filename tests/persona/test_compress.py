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
        short_result = "圧縮後テキスト" * 10
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=short_result)):
            result = compress_heavy_to_light(heavy)
        assert isinstance(result, str)
        assert result == short_result

    def test_normal_short_text(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        heavy = "短い重量ブロック" * 5
        compressed = "圧縮後"
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=compressed)) as mock_call:
            result = compress_heavy_to_light(heavy)
        mock_call.assert_called_once()
        assert result == compressed

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
        with patch("ghdag.llm.call", return_value=_make_llm_result()) as mock_call:
            compress_heavy_to_light("テスト", engine="claude", model="claude-haiku-4-5")
        _, kwargs = mock_call.call_args
        assert kwargs.get("engine") == "claude"
        assert kwargs.get("model") == "claude-haiku-4-5"

    def test_timeout_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("ghdag.llm.call", return_value=_make_llm_result()) as mock_call:
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
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="新しい軽量テキスト")):
            result = regenerate_light_block(persona_file)
        assert result.is_first_generation is True
        assert result.old_hash == ""
        assert result.light_text == "新しい軽量テキスト"
        content = persona_file.read_text(encoding="utf-8")
        assert "新しい軽量テキスト" in content

    def test_regeneration_changed(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="全く新しい軽量テキスト")):
            result = regenerate_light_block(persona_file)
        assert result.changed is True
        assert result.old_hash != result.new_hash
        content = persona_file.read_text(encoding="utf-8")
        assert "全く新しい軽量テキスト" in content

    def test_regeneration_unchanged(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        existing_light = "既存の軽量テキスト。"
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=existing_light)):
            result = regenerate_light_block(persona_file)
        assert result.changed is False
        assert result.old_hash == result.new_hash

    def test_drift_warning_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="変更後の軽量テキスト")):
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
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="新軽量テキスト")):
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
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="軽量テキスト")):
            result = regenerate_light_block(persona_file)
        assert result.persona_name == "テスト太郎"


# ---------------------------------------------------------------------------
# AC-4: 統合テスト
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_file_has_light_block_under_400_chars(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block, _split_h2_blocks
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        compressed = "あ" * 200
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout=compressed)):
            regenerate_light_block(persona_file)
        content = persona_file.read_text(encoding="utf-8")
        from mltgnt.persona.frontmatter import split_yaml_frontmatter
        _, body = split_yaml_frontmatter(content)
        blocks = _split_h2_blocks(body)
        light_text = blocks.get("軽量", "")
        assert len(light_text) <= 400

    def test_loader_compatible_after_regeneration(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        from mltgnt.persona.loader import load
        persona_file = tmp_path / "テスト太郎.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("ghdag.llm.call", return_value=_make_llm_result(stdout="軽量テキスト")):
            regenerate_light_block(persona_file)
        persona = load(persona_file)
        assert persona.name == "テスト太郎"
