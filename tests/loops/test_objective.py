"""tests/loops/test_objective.py — Objective 入力契約テスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltgnt.loops.objective import Objective, ObjectiveError, parse_objective


def _write_objective(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


@patch("mltgnt.loops.objective.md_read")
def test_id_omitted_uses_stem(mock_read, tmp_path):
    mock_read.return_value = MagicMock(
        frontmatter={},
        content="First line title\nRest of body",
    )
    path = _write_objective(tmp_path, "hp-renewal.md", "")
    result = parse_objective(path, default_persona="mizuho", default_max_iterations=5)
    assert isinstance(result, Objective)
    assert result.loop_id == "hp-renewal"
    assert result.title == "First line title"
    assert result.agent == "mizuho"
    assert result.max_iterations == 5


@patch("mltgnt.loops.objective.md_read")
def test_invalid_id_format(mock_read, tmp_path):
    mock_read.return_value = MagicMock(frontmatter={"id": "../bad"}, content="body")
    path = _write_objective(tmp_path, "x.md", "")
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, ObjectiveError)


@patch("mltgnt.loops.objective.md_read")
def test_empty_body(mock_read, tmp_path):
    mock_read.return_value = MagicMock(frontmatter={}, content="   ")
    path = _write_objective(tmp_path, "x.md", "")
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, ObjectiveError)
    assert "empty" in result.message


@patch("mltgnt.loops.objective.md_read")
def test_max_iterations_bool(mock_read, tmp_path):
    mock_read.return_value = MagicMock(frontmatter={"max_iterations": True}, content="body")
    path = _write_objective(tmp_path, "x.md", "")
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, ObjectiveError)


@patch("mltgnt.loops.objective.md_read")
def test_max_iterations_out_of_range(mock_read, tmp_path):
    mock_read.return_value = MagicMock(frontmatter={"max_iterations": 11}, content="body")
    path = _write_objective(tmp_path, "x.md", "")
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, ObjectiveError)


@patch("mltgnt.loops.objective.md_read")
def test_duplicate_id(mock_read, tmp_path):
    mock_read.return_value = MagicMock(frontmatter={"id": "dup"}, content="body")
    path = _write_objective(tmp_path, "x.md", "")
    result = parse_objective(
        path, default_persona="m", default_max_iterations=5, known_ids={"dup"}
    )
    assert isinstance(result, ObjectiveError)
