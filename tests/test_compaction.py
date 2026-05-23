"""tests/test_compaction.py — extract_promote_candidates の単体テスト。

設計: Issue #996
"""
from __future__ import annotations

from mltgnt.memory._compaction import (
    CompactionResult,
    PromoteCandidate,
    extract_promote_candidates,
)
from mltgnt.memory._format import MemoryEntry


def _entry(source_tag: str, content: str, i: int = 0) -> MemoryEntry:
    ts = f"2026-05-{10 + i:02d}T00:00:00+09:00"
    return MemoryEntry(timestamp=ts, role="user", content=content, source_tag=source_tag)


# ---------------------------------------------------------------------------
# extract_promote_candidates
# ---------------------------------------------------------------------------


class TestExtractPromoteCandidates:
    def test_empty_entries_returns_empty(self):
        assert extract_promote_candidates([], min_recurrence=3) == []

    def test_single_topic_meets_threshold(self):
        entries = [_entry("project_x", f"context {i}", i) for i in range(3)]
        result = extract_promote_candidates(entries, min_recurrence=3)
        assert len(result) == 1
        cand = result[0]
        assert isinstance(cand, PromoteCandidate)
        assert cand.topic == "project_x"
        assert cand.recurrence == 3
        assert cand.source_entries == 3

    def test_below_threshold_returns_empty(self):
        entries = [_entry("project_x", f"content {i}", i) for i in range(2)]
        assert extract_promote_candidates(entries, min_recurrence=3) == []

    def test_high_threshold_returns_empty(self):
        entries = [_entry("project_x", f"c {i}", i) for i in range(10)]
        assert extract_promote_candidates(entries, min_recurrence=100) == []

    def test_two_topics_only_qualifying_returned(self):
        entries_a = [_entry("topic_a", f"a {i}", i) for i in range(4)]
        entries_b = [_entry("topic_b", f"b {i}", i) for i in range(2)]
        result = extract_promote_candidates(entries_a + entries_b, min_recurrence=3)
        assert len(result) == 1
        assert result[0].topic == "topic_a"

    def test_both_topics_qualify(self):
        entries_a = [_entry("topic_a", f"a {i}", i) for i in range(3)]
        entries_b = [_entry("topic_b", f"b {i}", i) for i in range(3)]
        result = extract_promote_candidates(entries_a + entries_b, min_recurrence=3)
        topics = {c.topic for c in result}
        assert "topic_a" in topics
        assert "topic_b" in topics

    def test_summary_contains_content(self):
        entries = [_entry("project_x", f"detail_{i}", i) for i in range(3)]
        result = extract_promote_candidates(entries, min_recurrence=3)
        assert len(result) == 1
        for i in range(3):
            assert f"detail_{i}" in result[0].summary

    def test_no_md_promote_call(self, monkeypatch):
        """extract_promote_candidates 内で md_promote は呼ばれない。"""
        called = []
        monkeypatch.setattr(
            "mltgnt.memory._compaction.extract_promote_candidates",
            lambda *a, **kw: called.append(True) or [],
            raising=False,
        )
        # 直接 import した関数を呼ぶのでモンキーパッチは不要
        # md_promote が存在しないことだけ確認する
        import mltgnt.memory._compaction as mod
        assert not hasattr(mod, "md_promote"), "md_promote が _compaction に混入している"


# ---------------------------------------------------------------------------
# CompactionResult に promote_candidates フィールドがあること
# ---------------------------------------------------------------------------


class TestCompactionResultField:
    def test_promote_candidates_field_exists(self):
        result = CompactionResult(
            before_bytes=100,
            after_bytes=50,
            summary="ok",
        )
        assert hasattr(result, "promote_candidates")
        assert result.promote_candidates == []

    def test_promote_candidates_set_explicitly(self):
        cand = PromoteCandidate(
            topic="test_topic",
            summary="summary text",
            source_entries=3,
            recurrence=3,
        )
        result = CompactionResult(
            before_bytes=100,
            after_bytes=50,
            summary="ok",
            promote_candidates=[cand],
        )
        assert len(result.promote_candidates) == 1
        assert result.promote_candidates[0].topic == "test_topic"
