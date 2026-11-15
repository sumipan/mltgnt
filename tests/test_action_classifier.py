"""tests/test_action_classifier.py — ActionClassifier acceptance tests (#1777)"""
from __future__ import annotations

from pathlib import Path

import pytest

from mltgnt.agent.action_classifier import ActionClass, ActionClassifier


def test_classify_unknown_tool_defaults_to_needs_review(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    assert classifier.classify("gh_pr_create", {}) == ActionClass.NEEDS_REVIEW


def test_classify_becomes_safe_after_threshold_approvals(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    for _ in range(5):
        classifier.record_feedback("write_diary", user_approved=True)

    assert classifier.classify("write_diary", {}) == ActionClass.SAFE


def test_classify_unknown_tool_name_returns_needs_review(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    assert classifier.classify("never_seen_tool", {}) == ActionClass.NEEDS_REVIEW


def test_rejection_prevents_safe_promotion(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    classifier.record_feedback("tool_x", user_approved=False)

    assert classifier.classify("tool_x", {}) == ActionClass.NEEDS_REVIEW


def test_below_threshold_stays_needs_review(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    for _ in range(4):
        classifier.record_feedback("tool_y", user_approved=True)

    assert classifier.classify("tool_y", {}) == ActionClass.NEEDS_REVIEW


def test_init_with_nonexistent_rules_path(tmp_path: Path) -> None:
    rules_path = tmp_path / "nonexistent.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    assert classifier.classify("any_tool", {}) == ActionClass.NEEDS_REVIEW


def test_persistence_and_reload(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.jsonl"
    classifier = ActionClassifier(rules_path=rules_path)

    for _ in range(5):
        classifier.record_feedback("write_diary", user_approved=True)

    reloaded = ActionClassifier(rules_path=rules_path)
    assert reloaded.classify("write_diary", {}) == ActionClass.SAFE



