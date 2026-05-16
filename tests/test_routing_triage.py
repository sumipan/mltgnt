"""Tests for mltgnt.routing.triage — extract_json_object / extract_triage_section / prepare_profile_for_triage."""
from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

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
    """## 軽量 セクションの本文を返す。"""
    md = "## 軽量\n内容"
    result = extract_triage_section(md)
    assert result == "内容"


def test_extract_triage_section_v1_fallback():
    """## 軽量 がなく ## トリアージ用 がある場合、v1 フォールバックでセクション本文を返す。"""
    import textwrap
    md = textwrap.dedent("""\
        ## トリアージ用

        トリアージ内容

        ## 基本情報

        内容
    """)
    result = extract_triage_section(md)
    assert result is not None
    assert "トリアージ内容" in result


def test_extract_triage_section_none_when_missing():
    """該当セクションがない markdown を渡すと None を返す。"""
    md = "## 基本情報\n内容のみ"
    assert extract_triage_section(md) is None


def test_extract_triage_section_v2_wins_over_v1():
    """## 軽量 と ## トリアージ用 が両方存在する場合、## 軽量 が優先される。"""
    import textwrap
    md = textwrap.dedent("""\
        ## 軽量

        v2内容

        ## トリアージ用

        v1内容
    """)
    result = extract_triage_section(md)
    assert result is not None
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
    """TRIAGE_PROFILE_MAX_CHARS を超える文字列は末尾省略メッセージ付きで切り詰められる。"""
    mock_logger = MagicMock()
    long_text = "a" * (TRIAGE_PROFILE_MAX_CHARS + 100)
    result = prepare_profile_for_triage(long_text, mock_logger)
    assert result is not None
    assert len(result) > TRIAGE_PROFILE_MAX_CHARS  # 省略メッセージ込み
    assert "省略" in result


def test_prepare_profile_short_text_not_truncated():
    """TRIAGE_PROFILE_MAX_CHARS 以内の文字列はそのまま返す。"""
    mock_logger = MagicMock()
    text = "短いテキスト"
    result = prepare_profile_for_triage(text, mock_logger)
    assert result is not None
    assert "省略" not in result
