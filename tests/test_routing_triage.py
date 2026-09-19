"""Tests for mltgnt.routing.triage — extract_json_object / extract_triage_section / prepare_profile_for_triage."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock


from mltgnt.routing.triage import (
    TRIAGE_PROFILE_MAX_CHARS,
    extract_json_object,
    extract_triage_section,
    prepare_profile_for_triage,
)

logger = logging.getLogger(__name__)


def _triage_heading(index: int) -> str:
    patterns = [
        value
        for value in extract_triage_section.__code__.co_consts
        if isinstance(value, str) and value.startswith("^##")
    ]
    return patterns[index].split(r"\s+")[1].split(r"\s*")[0]


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_empty():
    assert extract_json_object("") is None


def test_extract_json_object_valid():
    assert extract_json_object('{"mode": "direct"}') == {"mode": "direct"}


def test_extract_json_object_with_fence():
    assert extract_json_object('```json\n{"mode": "delegate"}\n```') == {"mode": "delegate"}


def test_extract_json_object_invalid():
    assert extract_json_object("{not json}") is None


def test_extract_json_object_surrounded():
    assert extract_json_object('prefix {"k": "v"} suffix') == {"k": "v"}


def test_extract_json_object_not_json():
    assert extract_json_object("not json") is None


# ---------------------------------------------------------------------------
# extract_triage_section
# ---------------------------------------------------------------------------


def test_extract_triage_section_v2():
    """Returns the body of the lightweight triage section (product header is Japanese)."""
    md = f"## {_triage_heading(0)}\ncontent"
    result = extract_triage_section(md)
    assert result == "content"


def test_extract_triage_section_v1_fallback():
    """Falls back to the v1 triage section when the v2 lightweight header is absent."""
    md = f"## {_triage_heading(1)}\n\ntriage content\n\n## Other\n\ncontent"
    result = extract_triage_section(md)
    assert result is not None
    assert "triage content" in result


def test_extract_triage_section_none_when_missing():
    """Returns None when no triage section is present."""
    md = "## Basic information\ncontent only"
    assert extract_triage_section(md) is None


def test_extract_triage_section_v2_wins_over_v1():
    """v2 lightweight section wins when both v1 and v2 triage headers exist."""
    md = (
        f"## {_triage_heading(0)}\n\nv2content\n\n"
        f"## {_triage_heading(1)}\n\nv1content"
    )
    result = extract_triage_section(md)
    assert result is not None
    assert "v2content" in result
    assert "v1content" not in result


# ---------------------------------------------------------------------------
# prepare_profile_for_triage
# ---------------------------------------------------------------------------


def test_prepare_profile_empty_string_returns_none():
    mock_logger = MagicMock()
    assert prepare_profile_for_triage("", mock_logger) is None


def test_prepare_profile_none_returns_none():
    mock_logger = MagicMock()
    assert prepare_profile_for_triage(None, mock_logger) is None


def test_prepare_profile_truncates_long_text():
    """TRIAGE_PROFILE_MAX_CHARS Strings over the limit are truncated with a suffix."""
    mock_logger = MagicMock()
    long_text = "a" * (TRIAGE_PROFILE_MAX_CHARS + 100)
    result = prepare_profile_for_triage(long_text, mock_logger)
    assert result is not None
    assert len(result) > TRIAGE_PROFILE_MAX_CHARS  # including truncation suffix
    assert "truncated" in result


def test_prepare_profile_short_text_not_truncated():
    """TRIAGE_PROFILE_MAX_CHARS You can check the string within the same time."""
    mock_logger = MagicMock()
    text = "Short text"
    result = prepare_profile_for_triage(text, mock_logger)
    assert result is not None
    assert "truncated" not in result
