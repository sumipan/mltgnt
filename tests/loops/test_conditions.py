"""tests/loops/test_conditions.py — path_exists / path_changed の決定論評価。"""
from __future__ import annotations

from pathlib import Path

from mltgnt.loops.conditions import MISSING_PATH_TOKEN, PathConditionEvaluator


def test_path_exists_satisfied_and_pending(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "ok.txt").write_text("x", encoding="utf-8")
    ev = PathConditionEvaluator(root)
    assert ev.evaluate({"type": "path_exists", "path": "ok.txt"}, previous_token=None).status == "satisfied"
    assert ev.evaluate({"type": "path_exists", "path": "missing.txt"}, previous_token=None).status == "pending"


def test_path_exists_rejects_absolute_and_escape(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    ev = PathConditionEvaluator(root)
    abs_v = ev.evaluate({"type": "path_exists", "path": "/etc/passwd"}, previous_token=None)
    assert abs_v.status == "failed"
    assert "absolute" in abs_v.detail
    esc = ev.evaluate({"type": "path_exists", "path": "../outside.txt"}, previous_token=None)
    assert esc.status == "failed"
    assert "escapes" in esc.detail


def test_path_changed_initial_same_and_modified(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "f.txt"
    target.write_text("v1", encoding="utf-8")
    ev = PathConditionEvaluator(root)
    first = ev.evaluate({"type": "path_changed", "path": "f.txt"}, previous_token=None)
    assert first.status == "pending"
    assert first.observed_token is not None
    same = ev.evaluate(
        {"type": "path_changed", "path": "f.txt"}, previous_token=first.observed_token
    )
    assert same.status == "pending"
    assert same.observed_token == first.observed_token
    target.write_text("v2", encoding="utf-8")
    changed = ev.evaluate(
        {"type": "path_changed", "path": "f.txt"}, previous_token=first.observed_token
    )
    assert changed.status == "satisfied"


def test_path_changed_missing_to_created(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    ev = PathConditionEvaluator(root)
    first = ev.evaluate({"type": "path_changed", "path": "later.txt"}, previous_token=None)
    assert first.status == "pending"
    assert first.observed_token == MISSING_PATH_TOKEN
    (root / "later.txt").write_text("now", encoding="utf-8")
    second = ev.evaluate(
        {"type": "path_changed", "path": "later.txt"}, previous_token=MISSING_PATH_TOKEN
    )
    assert second.status == "satisfied"


def test_path_changed_rejects_directory(tmp_path: Path):
    root = tmp_path / "root"
    (root / "dir").mkdir(parents=True)
    ev = PathConditionEvaluator(root)
    v = ev.evaluate({"type": "path_changed", "path": "dir"}, previous_token=None)
    assert v.status == "failed"
    assert "director" in v.detail
