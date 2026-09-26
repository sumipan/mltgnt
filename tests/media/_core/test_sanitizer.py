"""mltgnt.media._core.sanitizer (#4029)."""

from __future__ import annotations

import json
import re

from mltgnt.media._core import sanitizer


def _line(obj: object) -> str:
    return json.dumps(obj)


def _assistant(text: str, **extra: object) -> str:
    return _line(
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}, **extra}
    )


def test_extract_final_assistant_text_after_last_tool_call() -> None:
    events = "\n".join(
        [
            _assistant("before"),
            _line({"type": "tool_call"}),
            _assistant("partial", timestamp_ms=1),
            _assistant("final"),
            "not json",
            _line([1, 2]),
            "",
        ]
    )
    assert sanitizer.extract_final_assistant_text(events) == "final"


def test_extract_final_assistant_text_none_cases() -> None:
    assert sanitizer.extract_final_assistant_text("") is None
    assert sanitizer.extract_final_assistant_text(_assistant("a") + "\n" + _line({"type": "tool_call"})) is None
    odd = [
        _line({"type": "assistant", "message": "x"}),
        _line({"type": "assistant", "message": {"role": "assistant", "content": "x"}}),
        _line({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "image"}, "x"]}}),
        _line({"type": "user"}),
    ]
    assert sanitizer.extract_final_assistant_text("\n".join(odd)) is None


def test_extract_final_assistant_text_without_tool_call() -> None:
    assert sanitizer.extract_final_assistant_text(_assistant("a") + "\n" + _assistant("b")) == "b"


def test_strip_status_lines_uses_given_patterns_only() -> None:
    text = "DONE\nkeep me\n  STATUS: ok\nSTATUS-ish line"
    assert sanitizer.strip_status_lines(text) == text.strip()
    assert sanitizer.strip_status_lines(text, prefixes=("STATUS:",), exact=("DONE",)) == ("keep me\nSTATUS-ish line")


def test_strip_leading_paragraphs() -> None:
    body = "\n\nWorking on it.\n\nStill working.\n\nThe answer is 42.\n\nWorking on it."
    is_noise = re.compile(r"(Working on it|Still working)\.").fullmatch
    assert sanitizer.strip_leading_paragraphs(body, lambda p: bool(is_noise(p.strip()))) == (
        "The answer is 42.\n\nWorking on it."
    )
    assert sanitizer.strip_leading_paragraphs("  ", lambda p: True) == ""


def test_dedupe_trailing_repeated_block() -> None:
    block = "x" * 90
    assert sanitizer.dedupe_trailing_repeated_block(f"head\n\n{block}\n\n{block}") == f"head\n\n{block}"
    short = "abc\n\nabc"
    assert sanitizer.dedupe_trailing_repeated_block(short) == short
    single = "y" * 200
    assert sanitizer.dedupe_trailing_repeated_block(single) == single
    different = f"{'a' * 90}\n\n{'b' * 90}"
    assert sanitizer.dedupe_trailing_repeated_block(different) == different


def test_sanitize_result_body_defaults_are_neutral() -> None:
    text = "  Answer body.\n\n<!-- internal -->\nSTATUS: ok  "
    assert sanitizer.sanitize_result_body(text) == "Answer body.\n\n<!-- internal -->\nSTATUS: ok"
    assert sanitizer.sanitize_result_body("   ") == ""


def test_sanitize_result_body_with_patterns() -> None:
    text = "Thinking.\n\nAnswer body.\nSTATUS: ok\n\n<!-- internal -->\ntrailer"
    result = sanitizer.sanitize_result_body(
        text,
        cut_markers=("<!-- internal", "\n\nnever present"),
        status_prefixes=("STATUS:",),
        leading_noise=lambda p: p.strip() == "Thinking.",
        formatter=lambda s: s.replace("Answer", "Final"),
    )
    assert result == "Final body."
