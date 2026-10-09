"""mltgnt.persona.extractor — H2 section parsing (#3318)."""
from __future__ import annotations

import dataclasses
import logging

import pytest

from mltgnt.config.language import EN
from mltgnt.persona.extractor import extract, parse_sections

_LIGHT_KEY = "Legacy light"
_BASIC_KEY = "Legacy background"
_HEAVY_KEY = "Legacy heavy"
_LEGACY_PACK = dataclasses.replace(
    EN,
    persona_section_aliases={
        _LIGHT_KEY: "Light",
        _BASIC_KEY: "Background",
        _HEAVY_KEY: "Heavy",
    },
)


@pytest.fixture(autouse=True)
def _legacy_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mltgnt.config.language._current", _LEGACY_PACK)


def test_parse_sections_basic() -> None:
    result = parse_sections("# Title\n\n## A\nfoo\n\n## B\nbar")
    assert result == {"A": "foo", "B": "bar"}


def test_parse_sections_empty() -> None:
    assert parse_sections("") == {}


def test_extract_light_light_section() -> None:
    sections = {_LIGHT_KEY: "light content", _HEAVY_KEY: "heavy content"}
    assert extract(sections, "light") == "light content"


def test_extract_light_fallback_basic_info() -> None:
    sections = {_BASIC_KEY: "basic info"}
    assert extract(sections, "light") == "basic info"


def test_extract_light_fallback_truncation() -> None:
    body = "x" * 600
    sections = {"Values": "some value"}
    assert extract(sections, "light", body=body) == body[:500]


def test_extract_light_warning_logged(caplog: pytest.LogCaptureFixture) -> None:
    body = "x" * 600
    with caplog.at_level(logging.WARNING, logger="mltgnt.persona.extractor"):
        extract(sections={}, mode="light", body=body, name="my_persona")
    assert "my_persona" in caplog.text


def test_extract_heavy_heavy_section() -> None:
    sections = {_LIGHT_KEY: "light content", _HEAVY_KEY: "heavy content here"}
    assert extract(sections, "heavy") == "heavy content here"


def test_extract_heavy_fallback() -> None:
    body = "full body content of old format persona"
    sections = {"Values": "some value"}
    assert extract(sections, "heavy", body=body) == body
