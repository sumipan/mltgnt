"""Tests for mltgnt.persona.compress — LLM compression, hashing, regeneration, and drift detection."""
from __future__ import annotations

import hashlib
import logging
import re
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.config.language import LanguagePack

# ---------------------------------------------------------------------------
# ASCII LanguagePack for validation tests
# ---------------------------------------------------------------------------

_ASCII_PACK = LanguagePack(
    work_request_markers=("please", "update", "revise"),
    create_request_markers=("create", "make"),
    deferred_patterns=(re.compile(r"later"),),
    compress_prompt_template="Generate a light block from: {heavy_text}",
    v21_required_sections=("**tone**", "**values**", "**positive-reaction**", "**friction**"),
    v21_example_section="**speech-example**",
    meta_header_needles=("as-persona",),
    dedupe_opener_re=re.compile(r"^plan[,]"),
    persona_cut_re=re.compile(r"\n\ntone-body"),
    exclude_stems=frozenset(),
)

# ---------------------------------------------------------------------------
# Persona file section header bytes
# (compressed.py uses hardcoded Japanese keys; store as bytes to keep this
# source file CJK-free while producing the required characters at runtime)
# ---------------------------------------------------------------------------

_SECT_LIGHT = b"\xe8\xbb\xbd\xe9\x87\x8f".decode()
_SECT_HEAVY = b"\xe9\x87\x8d\xe9\x87\x8f".decode()
_SECT_REF   = b"\xe5\x8f\x82\xe7\x85\xa7".decode()

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_llm_result(ok: bool = True, stdout: str = "compressed text", stderr: str = "") -> MagicMock:
    r = MagicMock()
    r.success = ok
    r.body = stdout
    r.stderr = stderr
    return r


# v2.1 mock response using ASCII LanguagePack sections
_V21_MOCK_RESPONSE = textwrap.dedent("""\
    persona-c is a curious and logical person who engages eagerly with new technology.

    **tone** — Speaks concisely and uses technical terms appropriately.
    **values** — Values efficiency and accuracy; dislikes ambiguity.
    **positive-reaction** — Engages eagerly with logical proposals and new tech topics.
    **friction** — Shows dissatisfaction with baseless claims or inefficient steps.
""")

# v2.1 form with speech examples
_V21_MOCK_WITH_SPEECH = textwrap.dedent("""\
    persona-c is a curious and logical person who engages eagerly with new technology.

    **tone** — Speaks concisely and uses technical terms appropriately.
    **values** — Values efficiency and accuracy; dislikes ambiguity.
    **positive-reaction** — Engages eagerly with logical proposals and new tech topics.
    **friction** — Shows dissatisfaction with baseless claims or inefficient steps.
    **speech-example**
    > Please organize the evidence a bit more before we discuss that.
""")

V2_PERSONA = textwrap.dedent(f"""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## {_SECT_LIGHT}

    Existing light text.

    ## {_SECT_HEAVY}

    Detailed persona text covering values, reaction patterns, and tone.
    This persona is very curious and likes trying new things.

    ## {_SECT_REF}

    Reference links and supplemental info.
""")

V2_PERSONA_EMPTY_LIGHT = textwrap.dedent(f"""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## {_SECT_LIGHT}

    ## {_SECT_HEAVY}

    Detailed persona text covering values, reaction patterns, and tone.

    ## {_SECT_REF}

    Reference links and supplemental info.
""")

V1_PERSONA = textwrap.dedent("""\
    ---
    persona:
      name: persona-d
    ops:
      engine: claude
      model: claude-sonnet-4-6
    ---

    ## basic-info

    v1-format persona. No heavy block.

    ## personality

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
            result = compress_heavy_to_light(heavy, pack=_ASCII_PACK)
        assert isinstance(result, str)
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_normal_short_text(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        heavy = "short heavy block" * 5
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            result = compress_heavy_to_light(heavy, pack=_ASCII_PACK)
        mock_call.assert_called_once()
        assert result == _V21_MOCK_RESPONSE.strip()

    def test_empty_input_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with pytest.raises(RuntimeError):
            compress_heavy_to_light("", pack=_ASCII_PACK)

    def test_llm_failure_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", side_effect=TimeoutError("timeout")):
            with pytest.raises(RuntimeError, match="timeout"):
                compress_heavy_to_light("heavy text", pack=_ASCII_PACK)

    def test_llm_ok_false_raises_runtime_error(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(ok=False, stderr="engine error")):
            with pytest.raises(RuntimeError, match="engine error"):
                compress_heavy_to_light("heavy text", pack=_ASCII_PACK)

    def test_engine_and_model_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("test", engine="claude", model="claude-haiku-4-5", pack=_ASCII_PACK)
        _, kwargs = mock_call.call_args
        assert kwargs.get("engine") == "claude"
        assert kwargs.get("model") == "claude-haiku-4-5"

    def test_timeout_passed_to_llm(self) -> None:
        from mltgnt.persona.compress import compress_heavy_to_light
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)) as mock_call:
            compress_heavy_to_light("test", timeout=60, pack=_ASCII_PACK)
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
            result = regenerate_light_block(persona_file, pack=_ASCII_PACK)
        assert result.is_first_generation is True
        assert result.old_hash == ""
        assert result.light_text == _V21_MOCK_RESPONSE.strip()
        content = persona_file.read_text(encoding="utf-8")
        assert "**tone**" in content

    def test_regeneration_changed(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file, pack=_ASCII_PACK)
        assert result.changed is True
        assert result.old_hash != result.new_hash
        content = persona_file.read_text(encoding="utf-8")
        assert "**tone**" in content

    def test_regeneration_unchanged(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        v2_persona_v21_light = (
            "---\n"
            "persona:\n"
            "  name: persona-d\n"
            "ops:\n"
            "  engine: claude\n"
            "  model: claude-sonnet-4-6\n"
            "---\n"
            "\n"
            f"## {_SECT_LIGHT}\n"
            "\n"
            "persona-c is a curious and logical person who engages eagerly with new technology.\n"
            "\n"
            "**tone** — Speaks concisely and uses technical terms appropriately.\n"
            "**values** — Values efficiency and accuracy; dislikes ambiguity.\n"
            "**positive-reaction** — Engages eagerly with logical proposals and new tech topics.\n"
            "**friction** — Shows dissatisfaction with baseless claims or inefficient steps.\n"
            "\n"
            f"## {_SECT_HEAVY}\n"
            "\n"
            "Detailed persona text covering values, reaction patterns, and tone.\n"
            "This persona is very curious and likes trying new things.\n"
            "\n"
            f"## {_SECT_REF}\n"
            "\n"
            "Reference links and supplemental info.\n"
        )
        persona_file.write_text(v2_persona_v21_light, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file, pack=_ASCII_PACK)
        assert result.changed is False
        assert result.old_hash == result.new_hash

    def test_drift_warning_logged(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            with caplog.at_level(logging.WARNING, logger="mltgnt.persona.compress"):
                result = regenerate_light_block(persona_file, pack=_ASCII_PACK)
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
            regenerate_light_block(persona_file, pack=_ASCII_PACK)
        content = persona_file.read_text(encoding="utf-8")
        assert "persona:" in content
        assert "name: persona-d" in content
        assert "Detailed persona text" in content
        assert f"## {_SECT_HEAVY}" in content
        assert "Reference links and supplemental info" in content
        assert f"## {_SECT_REF}" in content

    def test_invalid_v2_raises_value_error(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V1_PERSONA, encoding="utf-8")
        with pytest.raises(ValueError):
            regenerate_light_block(persona_file, pack=_ASCII_PACK)

    def test_result_persona_name(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            result = regenerate_light_block(persona_file, pack=_ASCII_PACK)
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
            regenerate_light_block(persona_file, pack=_ASCII_PACK)
        content = persona_file.read_text(encoding="utf-8")
        from mltgnt.persona.frontmatter import split_yaml_frontmatter
        _, body = split_yaml_frontmatter(content)
        blocks = _split_h2_blocks(body)
        light_text = blocks.get(_SECT_LIGHT, "")
        assert len(light_text) <= 1500

    def test_loader_compatible_after_regeneration(self, tmp_path: Path) -> None:
        from mltgnt.persona.compress import regenerate_light_block
        from mltgnt.persona.loader import load
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=_V21_MOCK_RESPONSE)):
            regenerate_light_block(persona_file, pack=_ASCII_PACK)
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
        _validate_v21_light_block(_V21_MOCK_RESPONSE, pack=_ASCII_PACK)

    def test_valid_with_speech_examples(self) -> None:
        """Valid: v2.1 block with speech examples raises no error."""
        from mltgnt.persona.compress import _validate_v21_light_block
        _validate_v21_light_block(_V21_MOCK_WITH_SPEECH, pack=_ASCII_PACK)

    def test_error_no_lead_text(self) -> None:
        """Invalid: no lead text (starts directly at first bold heading) -> ValueError."""
        from mltgnt.persona.compress import _validate_v21_light_block
        no_lead = (
            "**tone** — Speaks concisely.\n"
            "**values** — Values efficiency.\n"
            "**positive-reaction** — Likes logical proposals.\n"
            "**friction** — Dislikes baseless claims.\n"
        )
        with pytest.raises(ValueError, match="lead text"):
            _validate_v21_light_block(no_lead, pack=_ASCII_PACK)

    def test_error_missing_section_tone(self) -> None:
        """Invalid: missing **tone** -> ValueError mentions tone."""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = (
            "persona-c is a curious person.\n\n"
            "**values** — Values efficiency.\n"
            "**positive-reaction** — Likes logical proposals.\n"
            "**friction** — Dislikes baseless claims.\n"
        )
        with pytest.raises(ValueError, match=r"\*\*tone\*\*"):
            _validate_v21_light_block(missing_section, pack=_ASCII_PACK)

    def test_error_missing_section_values(self) -> None:
        """Invalid: missing **values** -> ValueError mentions values."""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = (
            "persona-c is a curious person.\n\n"
            "**tone** — Speaks concisely.\n"
            "**positive-reaction** — Likes logical proposals.\n"
            "**friction** — Dislikes baseless claims.\n"
        )
        with pytest.raises(ValueError, match=r"\*\*values\*\*"):
            _validate_v21_light_block(missing_section, pack=_ASCII_PACK)

    def test_error_missing_section_positive_reaction(self) -> None:
        """Invalid: missing **positive-reaction** -> ValueError mentions positive-reaction."""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = (
            "persona-c is a curious person.\n\n"
            "**tone** — Speaks concisely.\n"
            "**values** — Values efficiency.\n"
            "**friction** — Dislikes baseless claims.\n"
        )
        with pytest.raises(ValueError, match=r"\*\*positive-reaction\*\*"):
            _validate_v21_light_block(missing_section, pack=_ASCII_PACK)

    def test_error_missing_section_friction(self) -> None:
        """Invalid: missing **friction** -> ValueError mentions friction."""
        from mltgnt.persona.compress import _validate_v21_light_block
        missing_section = (
            "persona-c is a curious person.\n\n"
            "**tone** — Speaks concisely.\n"
            "**values** — Values efficiency.\n"
            "**positive-reaction** — Likes logical proposals.\n"
        )
        with pytest.raises(ValueError, match=r"\*\*friction\*\*"):
            _validate_v21_light_block(missing_section, pack=_ASCII_PACK)

    def test_error_speech_example_without_quote(self) -> None:
        """Invalid: **speech-example** without a following > line -> ValueError."""
        from mltgnt.persona.compress import _validate_v21_light_block
        bad_speech = (
            "persona-c is a curious person.\n\n"
            "**tone** — Speaks concisely.\n"
            "**values** — Values efficiency.\n"
            "**positive-reaction** — Likes logical proposals.\n"
            "**friction** — Dislikes baseless claims.\n"
            "**speech-example**\n"
            "Please organize the evidence a bit more.\n"
        )
        with pytest.raises(ValueError):
            _validate_v21_light_block(bad_speech, pack=_ASCII_PACK)


# ---------------------------------------------------------------------------
# regenerate_light_block invokes v2.1 validation
# ---------------------------------------------------------------------------


class TestRegenerateLightBlockV21Validation:
    def test_invalid_v21_raises_value_error(self, tmp_path: Path) -> None:
        """ValueError when LLM returns a response that fails validation."""
        from mltgnt.persona.compress import regenerate_light_block
        persona_file = tmp_path / "persona-d.md"
        persona_file.write_text(V2_PERSONA_EMPTY_LIGHT, encoding="utf-8")
        bad_response = (
            "**tone** — No lead text.\n"
            "**values** — Efficiency.\n"
            "**positive-reaction** — OK.\n"
            "**friction** — NG."
        )
        with patch("mltgnt.bridges.llm_adapter.call_llm", return_value=_make_llm_result(stdout=bad_response)):
            with pytest.raises(ValueError, match="lead text"):
                regenerate_light_block(persona_file, pack=_ASCII_PACK)
