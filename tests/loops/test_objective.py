"""tests/loops/test_objective.py — Objective 入力契約テスト。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from mltgnt.loops.objective import (
    Objective,
    ObjectiveError,
    ensure_frontmatter,
    parse_objective,
)


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


def test_ensure_frontmatter_completes_missing_keys(tmp_path):
    path = _write_objective(tmp_path, "foo.md", "# Foo\n\nbody\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    text = path.read_text(encoding="utf-8")
    assert "id: foo" in text
    assert "title: Foo" in text
    assert "status: active" in text
    assert "max_iterations: 5" in text
    assert "agent:" not in text
    assert "# Foo" in text


def test_ensure_frontmatter_title_strips_heading_markers(tmp_path):
    path = _write_objective(tmp_path, "h2.md", "## 試し\n\nbody\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    text = path.read_text(encoding="utf-8")
    assert "title: 試し" in text
    assert "'## 試し'" not in text
    assert "\n## 試し" in text  # 本文の見出しはそのまま


def test_ensure_frontmatter_title_deep_heading_and_no_space(tmp_path):
    path = _write_objective(tmp_path, "h6.md", "###### Deep\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    assert "title: Deep" in path.read_text(encoding="utf-8")

    # `#` 直後に空白がない行は ATX 見出しではないのでそのまま
    path2 = _write_objective(tmp_path, "tag.md", "#hashtag line\n")
    assert ensure_frontmatter(path2, default_max_iterations=5) is True
    assert "title: '#hashtag line'" in path2.read_text(encoding="utf-8")


def test_ensure_frontmatter_noop_when_complete(tmp_path):
    content = "---\nid: foo\ntitle: Foo\nstatus: active\nmax_iterations: 3\n---\n# Foo\n"
    path = _write_objective(tmp_path, "foo.md", content)
    mtime_before = path.stat().st_mtime
    assert ensure_frontmatter(path, default_max_iterations=5) is False
    assert path.read_text(encoding="utf-8") == content
    assert path.stat().st_mtime == mtime_before


def test_ensure_frontmatter_non_ascii_stem_hashes(tmp_path):
    path = _write_objective(tmp_path, "日本語.md", "plain title\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    text = path.read_text(encoding="utf-8")
    assert "id: objective-" in text
    assert "title: plain title" in text


def test_ensure_frontmatter_long_stem_truncated(tmp_path):
    stem = "a" * 80
    path = _write_objective(tmp_path, f"{stem}.md", "# Title\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, Objective)
    assert len(result.loop_id) <= 64
    assert result.loop_id == "a" * 64


def test_ensure_frontmatter_empty_body_still_completes(tmp_path):
    path = _write_objective(tmp_path, "empty.md", "")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, ObjectiveError)
    assert "empty" in result.message


def test_ensure_frontmatter_broken_yaml_no_write(tmp_path):
    content = "---\nid: [unterminated\n---\nbody\n"
    path = _write_objective(tmp_path, "broken.md", content)
    assert ensure_frontmatter(path, default_max_iterations=5) is False
    assert path.read_text(encoding="utf-8") == content


def test_persona_priority_agent_over_default(tmp_path):
    path = _write_objective(
        tmp_path,
        "p.md",
        "---\nid: p\ntitle: P\nstatus: active\nmax_iterations: 5\nagent: ando\n---\nbody\n",
    )
    result = parse_objective(path, default_persona="mizuho", default_max_iterations=5)
    assert isinstance(result, Objective)
    assert result.agent == "ando"


def test_persona_priority_default_when_agent_absent(tmp_path):
    path = _write_objective(
        tmp_path,
        "p.md",
        "---\nid: p\ntitle: P\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    result = parse_objective(path, default_persona="from-request", default_max_iterations=5)
    assert isinstance(result, Objective)
    assert result.agent == "from-request"


def test_plan_approval_true_false_default_and_invalid(tmp_path):
    path = _write_objective(
        tmp_path,
        "p.md",
        "---\nid: p\ntitle: P\nstatus: active\nmax_iterations: 5\nplan_approval: true\n---\nbody\n",
    )
    result = parse_objective(path, default_persona="m", default_max_iterations=5)
    assert isinstance(result, Objective)
    assert result.plan_approval is True

    path2 = _write_objective(
        tmp_path,
        "p2.md",
        "---\nid: p2\ntitle: P\nstatus: active\nmax_iterations: 5\nplan_approval: false\n---\nbody\n",
    )
    result2 = parse_objective(path2, default_persona="m", default_max_iterations=5)
    assert isinstance(result2, Objective)
    assert result2.plan_approval is False

    path3 = _write_objective(
        tmp_path,
        "p3.md",
        "---\nid: p3\ntitle: P\nstatus: active\nmax_iterations: 5\n---\nbody\n",
    )
    result3 = parse_objective(
        path3, default_persona="m", default_max_iterations=5, plan_approval_default=True
    )
    assert isinstance(result3, Objective)
    assert result3.plan_approval is True

    path4 = _write_objective(
        tmp_path,
        "p4.md",
        "---\nid: p4\ntitle: P\nstatus: active\nmax_iterations: 5\nplan_approval: maybe\n---\nbody\n",
    )
    result4 = parse_objective(path4, default_persona="m", default_max_iterations=5)
    assert isinstance(result4, ObjectiveError)
    assert "plan_approval" in result4.message


def test_ensure_frontmatter_does_not_add_plan_approval(tmp_path):
    path = _write_objective(tmp_path, "foo.md", "# Foo\n\nbody\n")
    assert ensure_frontmatter(path, default_max_iterations=5) is True
    assert "plan_approval" not in path.read_text(encoding="utf-8")
