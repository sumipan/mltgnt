"""mltgnt.persona.formatter — 口調整形（#3318）。"""
from __future__ import annotations

from pathlib import Path

from mltgnt.persona.formatter import (
    dedupe_persona_prefix,
    extract_persona_block_after_meta_headers,
    format_persona_body,
)


def test_extract_persona_block_after_meta_headers() -> None:
    raw = (
        "前置き\n\n"
        "あんどぅーとしての応答（標準出力相当）\n\n"
        "んー、わかるかも。\n\n"
        "---\n\n（以上）\n"
    )
    out = extract_persona_block_after_meta_headers(raw)
    assert out.startswith("んー、わかるかも。")
    assert "としての応答" not in out
    assert "（以上）" not in out


def test_dedupe_persona_prefix_keeps_last_opener() -> None:
    body = (
        "今週（3/24〜3/28）の計画、最初の要約。\n\n"
        "今週（3/24〜3/28）の計画、本当の本文だけ残す。"
    )
    out = dedupe_persona_prefix(body)
    assert out.startswith("今週（3/24〜3/28）の計画、本当の本文")
    assert out.count("今週（") == 1


def test_dedupe_persona_prefix_noop_when_single() -> None:
    body = "今週（3/24〜3/28）の計画、これだけ。"
    assert dedupe_persona_prefix(body) == body


def test_format_persona_body_composes_extract_and_dedupe() -> None:
    raw = (
        "メタ\n\n"
        "あんどぅーとしての応答（標準出力相当）\n\n"
        "今週（3/24〜3/28）の計画、一回目。\n\n"
        "今週（3/24〜3/28）の計画、二回目だけ残す。"
    )
    out = format_persona_body(raw)
    assert "としての応答" not in out
    assert out.count("今週（") == 1
    assert "二回目だけ残す" in out


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
