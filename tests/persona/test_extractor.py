"""mltgnt.persona.extractor — H2 セクション解析（#3318）。"""
from __future__ import annotations

import logging

import pytest

from mltgnt.persona.extractor import extract, parse_sections


def test_parse_sections_basic() -> None:
    result = parse_sections("# Title\n\n## A\nfoo\n\n## B\nbar")
    assert result == {"A": "foo", "B": "bar"}


def test_parse_sections_empty() -> None:
    assert parse_sections("") == {}


def test_extract_light_軽量_section() -> None:
    sections = {"軽量": "light content", "重量": "heavy content"}
    assert extract(sections, "light") == "light content"


def test_extract_light_fallback_基本情報() -> None:
    sections = {"基本情報": "basic info"}
    assert extract(sections, "light") == "basic info"


def test_extract_light_fallback_truncation() -> None:
    body = "x" * 600
    sections = {"価値観": "some value"}
    assert extract(sections, "light", body=body) == body[:500]


def test_extract_light_warning_logged(caplog: pytest.LogCaptureFixture) -> None:
    body = "x" * 600
    with caplog.at_level(logging.WARNING, logger="mltgnt.persona.extractor"):
        extract(sections={}, mode="light", body=body, name="my_persona")
    assert "my_persona" in caplog.text


def test_extract_heavy_重量_section() -> None:
    sections = {"軽量": "light content", "重量": "heavy content here"}
    assert extract(sections, "heavy") == "heavy content here"


def test_extract_heavy_fallback() -> None:
    body = "full body content of old format persona"
    sections = {"価値観": "some value"}
    assert extract(sections, "heavy", body=body) == body
