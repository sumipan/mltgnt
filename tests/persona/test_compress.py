"""Tests for mltgnt.persona.compress — LLM compression, hashing, regeneration, and drift detection."""
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

def _make_llm_result(ok: bool = True, stdout: str = "compressed text", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.success = ok
    r.body = stdout
    r.stderr = stderr
    return r


# Japanese text intentionally kept for CJK processing test
# v2.1 mock response (must pass _validate_v21_light_block)
_V21_MOCK_RESPONSE = textwrap.dedent("""\
    persona-c is a curious and logical person who engages eagerly with new technology.

    **\u53e3\u8abf** — Speaks concisely and uses technical terms appropriately.
    **\u4fa1\u5024\u89b3** — Values efficiency and accuracy; dislikes ambiguity.
    **\u597d\u610f\u7684\u53cd\u5fdc** — Engages eagerly with logical proposals and new tech topics.
    **\u5f15\u3063\u304b\u304b\u308b** — Shows dissatisfaction with baseless claims or inefficient steps.
""")

# Japanese text intentionally kept for CJK processing test
# v2.1 form with speech examples
_V21_MOCK_WITH_SPEECH = textwrap.dedent("""\
    persona-c is a curious and logical person who engages eagerly with new technology.

    **\u53e3\u8abf** — Speaks concisely and uses technical terms appropriately.
    **\u4fa1\u5024\u89b3** — Values efficiency and accuracy; dislikes ambiguity.
    **\u597d\u610f\u7684\u53cd\u5fdc** — Engages eagerly with logical proposals and new tech topics.
    **\u5f15\u3063\u304b\u304b\u308b** — Shows dissatisfaction with baseless claims or inefficient steps.
    **\u767a\u8a00\u4f8b**
    > Please organize the evidence a bit more before we discuss that.
""")

# Japanese text intentionally kept for CJK processing test
V2_PERSONA = textwrap.dedent("""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## \u8efd\u91cf

    Existing light text.

    ## \u91cd\u91cf

    Detailed persona text covering values, reaction patterns, and tone.
    This persona is very curious and likes trying new things.

    ## \u53c2\u7167

    Reference links and supplemental info.
""")

# Japanese text intentionally kept for CJK processing test
V2_PERSONA_EMPTY_LIGHT = textwrap.dedent("""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## \u8efd\u91cf

    ## \u91cd\u91cf

    Detailed persona text covering values, reaction patterns, and tone.

    ## \u53c2\u7167

    Reference links and supplemental info.
""")

# Japanese text intentionally kept for CJK processing test
V1_PERSONA = textwrap.dedent("""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## \u57fa\u672c\u60c5\u5831

    v1-format persona. No heavy block.

    ## \u4fa1\u5024\u89b3

    Likes testing.
""")


# ---------------------------------------------------------------------------
# AC-2: compute_block_hash
# ---------------------------------------------------------------------------


class TestComputeBlockHash:
    def test_returns_64char_hex(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        result = compute_block_hash("test string")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_strip_normalization(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        assert compute_block_hash("test string\n") == compute_block_hash("test string")

    def test_crlf_normalization(self) -> None:
        from mltgnt.persona.compress import compute_block_hash
        assert compute_block_hash("line1\r\nline2") == compute_block_hash("line1\nline2")

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
        heavy = "a" * 1500
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = compress_heavy_to_light(heavy)
        assert isinstance(result, str)
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_normal_short_text(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        heavy = "short heavy block" * 5
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            result = compress_heavy_to_light(heavy)
        mock_call.assert_called_once()
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_empty_input_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with pytest.raises(RuntimeError):
            compress_heavy_to_light("")

    def test_llm_failure_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=TimeoutError("timeout")):
            with pytest.raises(RuntimeError, match="timeout"):
                compress_heavy_to_light("heavy text")

    def test_llm_ok_false_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
            with pytest.raises(RuntimeError, match="engine error"):
                compress_heavy_to_light("heavy text")

    def test_engine_and_model_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("test", engine="claude", model="claude-haiku-4-5")
        _, kwargs = mock_call.call_args
        assert kwargs.get("engine") == "claude"
        assert kwargs.get("model") == "claude-haiku-4-5"

    def test_timeout_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("test", timeout=60)
        _, kwargs = mock_call.call_args
        assert kwargs.get("timeout") == 60


# ---------------------------------------------------------------------------
# AC-3: regenerate_light_block
# ---------------------------------------------------------------------------


class TestRegenerateLightBlock:
    def test_first_generation(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.is_first_generation is True
        assert result.old_hash == ""
        assert result.light_text == _V21_MOCK_RESPONSE.strip()
        content = persona_file.read_text(encoding="utf-8")
        # Japanese text intentionally kept for CJK processing test
        assert "**\u53e3\u8abf**" in content

    def test_regeneration_changed(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.changed is True
        assert result.old_hash != result.new_hash
        content = persona_file.read_text(encoding="utf-8")
        # Japanese text intentionally kept for CJK processing test
        assert "**\u53e3\u8abf**" in content

    def test_regeneration_unchanged(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        # Prepare persona with v2.1 light block already present
        # Japanese text intentionally kept for CJK processing test
        v2_persona_v21_light = """---
persona:
  name: persona-d
ops:
  engine: claude
  model: claude-sonnet-4-6
---

## \u8efd\u91cf

persona-c is a curious and logical person who engages eagerly with new technology.

**\u53e3\u8abf** — Speaks concisely and uses technical terms appropriately.
**\u4fa1\u5024\u89b3** — Values efficiency and accuracy; dislikes ambiguity.
**\u597d\u610f\u7684\u53cd\u5fdc** — Engages eagerly with logical proposals and new tech topics.
**\u5f15\u3063\u304b\u304b\u308b** — Shows dissatisfaction with baseless claims or inefficient steps.

## \u91cd\u91cf

Detailed persona text covering values, reaction patterns, and tone.
This persona is very curious and likes trying new things.

## \u53c2\u7167

Reference links and supplemental info.
"""
        persona_file.write_text(v2_persona_v21_light, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.changed is False
        assert result.old_hash == result.new_hash

    def test_drift_warning_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            with caplog.at_level(logging.WARNING, logger="mltgnt.persona.compress"):
                result = regenerate_light_block(persona_file)
        if result.changed:
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) > 0
            log_text = " ".join(r.message for r in warning_records)
            assert result.old_hash[:8] in log_text or result.new_hash[:8] in log_text

    def test_file_integrity(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        content = persona_file.read_text(encoding="utf-8")
        assert "persona:" in content
        assert "name: persona-d" in content
        assert "Detailed persona text" in content
        # Japanese text intentionally kept for CJK processing test
        assert "## \u91cd\u91cf" in content
        assert "Reference links and supplemental info" in content
        assert "## \u53c2\u7167" in content

    def test_invalid_v2_raises_value_error(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V1_PERSONA, encoding="utf-8")
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="v2 \u5f62\u5f0f\u3067\u306f\u3042\u308a\u307e\u305b\u3093"):
            regenerate_light_block(persona_file)

    def test_result_persona_name(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file)
        assert result.persona_name == "persona-d"


# ---------------------------------------------------------------------------
# AC-4: integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_file_has_light_block_under_1500_chars(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block, _split_h2_blocks
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        content = persona_file.read_text(encoding="utf-8")
        from mltgnt.persona.frontmatter import split_yaml_frontmatter
        _, body = split_yaml_frontmatter(content)
        blocks = _split_h2_blocks(body)
        # Japanese text intentionally kept for CJK processing test
        light_text = blocks.get("\u8efd\u91cf", "")
        assert len(light_text) <= 1500

    def test_loader_compatible_after_regeneration(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        from mltgnt.persona.loader import load
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file)
        persona = load(persona_file)
        assert persona.name == "persona-d"


# ---------------------------------------------------------------------------
# LIGHT_BLOCK_MAX_CHARS constant (AC-2 #1)
# ---------------------------------------------------------------------------


class TestLightBlockMaxChars:
    def test_constant_is_1500(self) -> None:
        from mltgnt.persona.compress import LIGHT_BLOCK_MAX_CHARS
        assert LIGHT_BLOCK_MAX_CHARS == 1500


# ---------------------------------------------------------------------------
# _validate_v21_light_block (AC-2 #5-7)
# ---------------------------------------------------------------------------


class TestValidateV21LightBlock:
    """Happy-path and error-path tests for _validate_v21_light_block."""

    def test_valid_standard_block(self) -> None:
        """Valid: standard v2.1 block raises no error."""
        from mltgnt.persona.compress import _validate_v21_light_block
        _validate_v21_light_block(_V21_MOCK_RESPONSE)

    def test_valid_with_speech_examples(self) -> None:
        """Valid: v2.1 block with speech examples raises no error."""
        from mltgnt.persona.compress import _validate_v21_light_block
        _validate_v21_light_block(_V21_MOCK_WITH_SPEECH)

    def test_error_no_lead_text(self) -> None:
        """Invalid: no lead text (starts at **\u53e3\u8abf**) → ValueError matching lead-text message."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        no_lead = """**\u53e3\u8abf** — Speaks concisely.
**\u4fa1\u5024\u89b3** — Values efficiency.
**\u597d\u610f\u7684\u53cd\u5fdc** — Likes logical proposals.
**\u5f15\u3063\u304b\u304b\u308b** — Dislikes baseless claims.
"""
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="lead text"):
            _validate_v21_light_block(no_lead)

    def test_error_missing_section_tone(self) -> None:
        """Invalid: missing **\u53e3\u8abf** → ValueError mentions \u53e3\u8abf."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        missing_section = """persona-c is a curious person.

**\u4fa1\u5024\u89b3** — Values efficiency.
**\u597d\u610f\u7684\u53cd\u5fdc** — Likes logical proposals.
**\u5f15\u3063\u304b\u304b\u308b** — Dislikes baseless claims.
"""
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="\u53e3\u8abf"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_values(self) -> None:
        """Invalid: missing **\u4fa1\u5024\u89b3** → ValueError mentions \u4fa1\u5024\u89b3."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        missing_section = """persona-c is a curious person.

**\u53e3\u8abf** — Speaks concisely.
**\u597d\u610f\u7684\u53cd\u5fdc** — Likes logical proposals.
**\u5f15\u3063\u304b\u304b\u308b** — Dislikes baseless claims.
"""
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="\u4fa1\u5024\u89b3"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_positive_reaction(self) -> None:
        """Invalid: missing **\u597d\u610f\u7684\u53cd\u5fdc** → ValueError mentions \u597d\u610f\u7684\u53cd\u5fdc."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        missing_section = """persona-c is a curious person.

**\u53e3\u8abf** — Speaks concisely.
**\u4fa1\u5024\u89b3** — Values efficiency.
**\u5f15\u3063\u304b\u304b\u308b** — Dislikes baseless claims.
"""
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="\u597d\u610f\u7684\u53cd\u5fdc"):
            _validate_v21_light_block(missing_section)

    def test_error_missing_section_friction(self) -> None:
        """Invalid: missing **\u5f15\u3063\u304b\u304b\u308b** → ValueError mentions \u5f15\u3063\u304b\u304b\u308b."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        missing_section = """persona-c is a curious person.

**\u53e3\u8abf** — Speaks concisely.
**\u4fa1\u5024\u89b3** — Values efficiency.
**\u597d\u610f\u7684\u53cd\u5fdc** — Likes logical proposals.
"""
        # Japanese text intentionally kept for CJK processing test
        with pytest.raises(ValueError, match="\u5f15\u3063\u304b\u304b\u308b"):
            _validate_v21_light_block(missing_section)

    def test_error_speech_example_without_quote(self) -> None:
        """Invalid: **\u767a\u8a00\u4f8b** without a following > line → ValueError."""
        from mltgnt.persona.compress import _validate_v21_light_block
        # Japanese text intentionally kept for CJK processing test
        bad_speech = """persona-c is a curious person.

**\u53e3\u8abf** — Speaks concisely.
**\u4fa1\u5024\u89b3** — Values efficiency.
**\u597d\u610f\u7684\u53cd\u5fdc** — Likes logical proposals.
**\u5f15\u3063\u304b\u304b\u308b** — Dislikes baseless claims.
**\u767a\u8a00\u4f8b**
Please organize the evidence a bit more.
"""
        with pytest.raises(ValueError):
            _validate_v21_light_block(bad_speech)


# ---------------------------------------------------------------------------
# regenerate_light_block invokes v2.1 validation
# ---------------------------------------------------------------------------


class TestRegenerateLightBlockV21Validation:
    def test_invalid_v21_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError when LLM returns a response that fails validation."""
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        # Japanese text intentionally kept for CJK processing test
        bad_response = "**\u53e3\u8abf** — No lead text.\n**\u4fa1\u5024\u89b3** — Efficiency.\n**\u597d\u610f\u7684\u53cd\u5fdc** — OK.\n**\u5f15\u3063\u304b\u304b\u308b** — NG."
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=bad_response)):
            # Japanese text intentionally kept for CJK processing test
            with pytest.raises(ValueError, match="lead text"):
                regenerate_light_block(persona_file)
