"""Tests for mltgnt.memory.core_render (Issue #4037)."""
from __future__ import annotations

import logging

import pytest

from mltgnt.memory.core_render import render_core
from mltgnt.memory.semantic import SemanticEntry


def _e(n: int, kind: str, content: str, *, status: str = "active", day: int = 1) -> SemanticEntry:
    return SemanticEntry(
        id=f"m_2026-09-{day:02d}_{n:04d}",
        ts=f"2026-09-{day:02d} 10:{n:02d}",
        kind=kind,
        content=content,
        subject="user",
        source="s",
        status=status,
    )


def test_empty_returns_empty_string() -> None:
    assert render_core([]) == ""
    assert render_core([_e(1, "fact", "x", status="superseded")]) == ""


def test_order_and_format() -> None:
    entries = [
        _e(1, "fact", "old fact"),
        _e(2, "fact", "new fact"),
        _e(3, "reflection", "a reflection"),
        _e(4, "self", "self note"),
        _e(5, "preference", "likes tea"),
        _e(6, "commitment", "send report"),
        _e(7, "caveat", "no calls"),
    ]
    assert render_core(entries).splitlines() == [
        "## Memory",
        "- [caveat] no calls",
        "- [commitment] send report",
        "- [preference] likes tea",
        "- [self] self note",
        "- [fact] new fact",
        "- [fact] old fact",
        "- [reflection] a reflection",
    ]


def test_superseded_entries_are_excluded() -> None:
    out = render_core([_e(1, "fact", "gone", status="superseded"), _e(2, "fact", "kept")])
    assert "gone" not in out
    assert "- [fact] kept" in out


def test_custom_heading_and_newlines_collapsed() -> None:
    out = render_core([_e(1, "fact", "line one\nline two")], heading="# Core")
    assert out == "# Core\n- [fact] line one line two"


def test_over_budget_drops_fact_and_preference_but_keeps_mandatory() -> None:
    entries = [_e(i, "fact", "f" * 200, day=2) for i in range(1, 30)]
    entries += [_e(i, "preference", "p" * 200, day=3) for i in range(1, 10)]
    entries += [_e(1, "caveat", "never share keys"), _e(2, "commitment", "ship on friday")]
    out = render_core(entries, max_bytes=4096)
    assert len(out.encode("utf-8")) <= 4096
    assert "- [caveat] never share keys" in out
    assert "- [commitment] ship on friday" in out
    assert out.count("[preference]") == 9
    assert 0 < out.count("[fact]") < 29


def test_newest_facts_win_under_budget() -> None:
    entries = [_e(1, "fact", "older " + "x" * 50), _e(2, "fact", "newer " + "y" * 50)]
    out = render_core(entries, max_bytes=80)
    assert "newer" in out
    assert "older" not in out


def test_mandatory_over_budget_is_not_truncated_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    entries = [_e(i, "caveat", "c" * 100) for i in range(1, 6)] + [_e(9, "fact", "dropped")]
    with caplog.at_level(logging.WARNING, logger="mltgnt.memory.core_render"):
        out = render_core(entries, max_bytes=200)
    assert out.count("[caveat]") == 5
    assert "dropped" not in out
    assert any("not truncated" in r.getMessage() for r in caplog.records)
