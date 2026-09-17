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
    # Japanese text intentionally kept for CJK processing test
    md = "## 軽量\n内容"
    result = extract_triage_section(md)
    # Japanese text intentionally kept for CJK processing test
    assert result == "内容"


def test_extract_triage_section_v1_fallback():
    """Falls back to the v1 triage section when the v2 lightweight header is absent."""
    # Japanese text intentionally kept for CJK processing test
    md = "## トリアージ用\n\nトリアージ内容\n\n## 基本情報\n\n内容"
    result = extract_triage_section(md)
    assert result is not None
    # Japanese text intentionally kept for CJK processing test
    assert "トリアージ内容" in result


def test_extract_triage_section_none_when_missing():
    """Returns None when no triage section is present."""
    # Japanese text intentionally kept for CJK processing test
    md = "## 基本情報\n内容のみ"
    assert extract_triage_section(md) is None


def test_extract_triage_section_v2_wins_over_v1():
    """v2 lightweight section wins when both v1 and v2 triage headers exist."""
    # Japanese text intentionally kept for CJK processing test
    md = "## 軽量\n\nv2内容\n\n## トリアージ用\n\nv1内容"
    result = extract_triage_section(md)
    assert result is not None
    # Japanese text intentionally kept for CJK processing test
    assert "v2内容" in result
    assert "v1内容" not in result


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
    # Japanese text intentionally kept for CJK processing test
    """TRIAGE_PROFILE_MAX_CHARS を超える文字列は末尾省略メッセージ付きで切り詰められる。"""
    mock_logger = MagicMock()
    long_text = "a" * (TRIAGE_PROFILE_MAX_CHARS + 100)
    result = prepare_profile_for_triage(long_text, mock_logger)
    assert result is not None
    # Japanese text intentionally kept for CJK processing test
    assert len(result) > TRIAGE_PROFILE_MAX_CHARS  # 省略メッセージ込み
    assert "省略" in result


def test_prepare_profile_short_text_not_truncated():
    """TRIAGE_PROFILE_MAX_CHARS You can check the string within the same time."""
    mock_logger = MagicMock()
    text = "Short text"
    result = prepare_profile_for_triage(text, mock_logger)
    assert result is not None
    # Japanese text intentionally kept for CJK processing test
    assert "省略" not in result
