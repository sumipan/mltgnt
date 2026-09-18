"""mltgnt.persona.formatter — tone formatting (#3318)."""
from __future__ import annotations

from pathlib import Path

from mltgnt.persona.formatter import (
    dedupe_persona_prefix,
    extract_persona_block_after_meta_headers,
    format_persona_body,
)

# Persona names that must not appear as literals in formatter source.
# Stored as bytes to keep this source file CJK-free while preserving the guard.
_PERSONA_NAME_A = b"\xe3\x81\x82\xe3\x82\x93\xe3\x81\xa9\xe3\x81\x85\xe3\x83\xbc".decode()
_PERSONA_NAME_B = b"\xe5\xae\x89\xe8\x97\xa4\xe7\x91\x9e\xe7\xa8\x80".decode()
_PERSONA_NAME_C = b"\xe3\x83\x8f\xe3\x83\x8b".decode()


def test_extract_persona_block_after_meta_headers(ascii_pack) -> None:
    raw = (
        "preamble\n\n"
        "persona-a as-persona (stdout-equivalent)\n\n"
        "hmm, I see.\n\n"
        "---\n\n"
    )
    out = extract_persona_block_after_meta_headers(raw, pack=ascii_pack)
    assert out.startswith("hmm, I see.")
    assert "as-persona" not in out


def test_dedupe_persona_prefix_keeps_last_opener(ascii_pack) -> None:
    body = (
        "this-week (W1) plan, first summary.\n\n"
        "this-week (W1) plan, keep only the real body."
    )
    out = dedupe_persona_prefix(body, pack=ascii_pack)
    assert out.startswith("this-week (W1) plan, keep only the real body")
    assert out.count("this-week (") == 1


def test_dedupe_persona_prefix_noop_when_single(ascii_pack) -> None:
    body = "this-week (W1) plan, just one."
    assert dedupe_persona_prefix(body, pack=ascii_pack) == body


def test_format_persona_body_composes_extract_and_dedupe(ascii_pack) -> None:
    raw = (
        "meta\n\n"
        "persona-a as-persona (stdout-equivalent)\n\n"
        "this-week (W1) plan, first pass.\n\n"
        "this-week (W1) plan, keep only this."
    )
    out = format_persona_body(raw, pack=ascii_pack)
    assert "as-persona" not in out
    assert out.count("this-week (") == 1
    assert "keep only this" in out


def test_format_persona_body_cuts_generic_tone_marker(ascii_pack) -> None:
    """Cut marker must match any persona token, not a host-specific name (#3337)."""
    raw = "body text.\n\npersona-a tone-body starts here with explanation"
    assert format_persona_body(raw, pack=ascii_pack) == "body text."


def test_formatter_source_has_no_media_or_sdk() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mltgnt"
        / "persona"
        / "formatter.py"
    ).read_text(encoding="utf-8")
    assert "jobs/" not in src
    assert "slack_sdk" not in src
    assert "ghdag" not in src
    assert "markdown_to_slack_mrkdwn" not in src
    assert _PERSONA_NAME_A not in src
    assert _PERSONA_NAME_B not in src
    assert _PERSONA_NAME_C not in src
