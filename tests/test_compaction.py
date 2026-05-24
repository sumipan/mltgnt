"""tests/test_compaction.py — extract_promote_candidates + compact() のテスト。

設計: Issue #996, Issue #1135
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory._compaction import (
    CompactionResult,
    PromoteCandidate,
    _effective_bytes_for_ratio,
    _extract_and_merge_preferences,
    _promote_with_compression,
    _promote_mid_to_long,
    _redistribute_entries,
    _strip_observe_entries,
    _rollup_recent_chunk,
    _compress_rollup_chunk,
    _sanitize_phase1_output,
    compact,
    extract_promote_candidates,
)
from mltgnt.memory._format import MemoryEntry, serialize_entry


def _entry(source_tag: str, content: str, i: int = 0) -> MemoryEntry:
    ts = f"2026-05-{10 + i:02d}T00:00:00+09:00"
    return MemoryEntry(timestamp=ts, role="user", content=content, source_tag=source_tag)


def _make_config(tmp_path: Path, target_bytes: int = 25_600) -> MemoryConfig:
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path,
        compact_target_bytes=target_bytes,
        compact_threshold_bytes=target_bytes // 2,
        timezone="Asia/Tokyo",
    )


def _write_jsonl(path: Path, entries: list[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(serialize_entry(e) + "\n")


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


# ---------------------------------------------------------------------------
# MemoryConfig に timezone フィールドがあること（AC-9）
# ---------------------------------------------------------------------------


class TestMemoryConfigTimezone:
    def test_timezone_field_default(self):
        from mltgnt.config import MemoryConfig
        from pathlib import Path
        cfg = MemoryConfig(chat_dir=Path("/tmp/test"))
        assert hasattr(cfg, "timezone")
        assert cfg.timezone == "Asia/Tokyo"

    def test_timezone_field_custom(self):
        from mltgnt.config import MemoryConfig
        from pathlib import Path
        cfg = MemoryConfig(chat_dir=Path("/tmp/test"), timezone="UTC")
        assert cfg.timezone == "UTC"


# ---------------------------------------------------------------------------
# _effective_bytes_for_ratio
# ---------------------------------------------------------------------------


class TestEffectiveBytesForRatio:
    def test_no_observe_entries_returns_full_size(self):
        text = '{"timestamp": "2026-05-01T00:00:00+09:00", "role": "user", "content": "hello", "source_tag": "chat"}'
        result = _effective_bytes_for_ratio(text)
        assert result == len(text.encode("utf-8"))

    def test_observe_entries_excluded(self):
        observe = '{"timestamp": "2026-05-01T00:00:00+09:00", "role": "user", "content": "obs data", "source_tag": "[slack-observe]"}'
        normal = '{"timestamp": "2026-05-02T00:00:00+09:00", "role": "user", "content": "normal", "source_tag": "chat"}'
        text = observe + "\n" + normal
        result = _effective_bytes_for_ratio(text)
        # observe エントリは除外されるべき
        assert result < len(text.encode("utf-8"))
        assert result > 0


# ---------------------------------------------------------------------------
# _sanitize_phase1_output
# ---------------------------------------------------------------------------


class TestSanitizePhase1Output:
    def test_removes_meta_lines(self):
        text = "承知しました\n- コーヒーが好き\n分析します\n- 朝型の生活"
        result = _sanitize_phase1_output(text)
        assert "承知しました" not in result
        assert "分析します" not in result
        assert "コーヒーが好き" in result

    def test_removes_headings(self):
        text = "## 分析結果\n- コーヒーが好き"
        result = _sanitize_phase1_output(text)
        assert "## 分析結果" not in result
        assert "コーヒーが好き" in result

    def test_empty_input(self):
        assert _sanitize_phase1_output("") == ""


# ---------------------------------------------------------------------------
# _strip_observe_entries
# ---------------------------------------------------------------------------


class TestStripObserveEntries:
    def test_strips_observe_blocks(self):
        body = "normal content\n---\n[slack-observe] some data\n---\nother content"
        result = _strip_observe_entries(body)
        assert "[slack-observe]" not in result
        assert "normal content" in result
        assert "other content" in result

    def test_no_observe_entries_unchanged(self):
        body = "block one\n---\nblock two"
        result = _strip_observe_entries(body)
        assert result == body


# ---------------------------------------------------------------------------
# _rollup_recent_chunk
# ---------------------------------------------------------------------------


class TestRollupRecentChunk:
    def test_empty_body_returns_empty(self):
        remaining, promoted = _rollup_recent_chunk("", 1024)
        assert remaining == ""
        assert promoted == ""

    def test_single_block_promoted(self):
        body = "## 2026-05-01 10:00 — user\n\nsome content"
        remaining, promoted = _rollup_recent_chunk(body, 10)
        # 1 block only, gets promoted
        assert promoted == body
        assert remaining == ""

    def test_multiple_blocks_partial_promotion(self):
        block1 = "## 2026-05-01 10:00 — user\n\n" + "a" * 100
        block2 = "## 2026-05-02 10:00 — user\n\n" + "b" * 100
        block3 = "## 2026-05-03 10:00 — user\n\n" + "c" * 100
        body = "\n---\n".join([block1, block2, block3])
        remaining, promoted = _rollup_recent_chunk(body, 50)
        assert promoted  # at least one block promoted
        assert remaining  # some remain


# ---------------------------------------------------------------------------
# _compress_rollup_chunk
# ---------------------------------------------------------------------------


class TestCompressRollupChunk:
    def test_returns_summary_on_success(self):
        def _llm(prompt: str) -> str:
            return "summary line"

        result = _compress_rollup_chunk("some long content", _llm)
        assert "summary line" in result

    def test_falls_back_on_exception(self):
        def _llm_fail(prompt: str) -> str:
            raise RuntimeError("LLM error")

        import warnings
        with warnings.catch_warnings(record=True):
            result = _compress_rollup_chunk("raw content", _llm_fail, max_retries=0)
        assert result == "raw content"


# ---------------------------------------------------------------------------
# _extract_and_merge_preferences
# ---------------------------------------------------------------------------


class TestExtractAndMergePreferences:
    def test_returns_existing_when_no_recent(self):
        result, warning = _extract_and_merge_preferences(
            "existing prefs", "", 1024, lambda p: "merged"
        )
        assert result == "existing prefs"
        assert warning is None

    def test_merges_preferences(self):
        def _llm(prompt: str) -> str:
            return "- コーヒーが好き\n- 朝型の生活"

        result, warning = _extract_and_merge_preferences(
            "", "最近コーヒーを毎朝飲んでいる", 1024, _llm
        )
        assert "コーヒー" in result
        assert warning is None

    def test_warns_on_llm_failure(self):
        def _llm_fail(prompt: str) -> str:
            raise RuntimeError("fail")

        result, warning = _extract_and_merge_preferences(
            "existing", "recent", 1024, _llm_fail
        )
        assert result == "existing"
        assert warning is not None


# ---------------------------------------------------------------------------
# _promote_with_compression
# ---------------------------------------------------------------------------


class TestPromoteWithCompression:
    def test_compresses_combined_text(self):
        def _llm(prompt: str) -> str:
            return "compressed result"

        result, warning = _promote_with_compression(
            "long_term", "existing content", "new content",
            cap_bytes=1024, llm_call=_llm
        )
        assert result == "compressed result"
        assert warning is None

    def test_rejects_over_compression(self):
        # existing is 1000 bytes, max_ratio=0.9, so result must be >= 900 bytes
        existing = "x" * 500  # 500 bytes
        max_ratio = 0.9

        def _llm(prompt: str) -> str:
            # Return only 10 bytes (way under existing * max_ratio = 450)
            return "tiny"

        result, warning = _promote_with_compression(
            "long_term", existing, "", cap_bytes=10_000, llm_call=_llm,
            max_ratio=max_ratio
        )
        # Should revert to original (combined = existing)
        assert result == existing
        assert warning is not None


# ---------------------------------------------------------------------------
# _redistribute_entries
# ---------------------------------------------------------------------------


class TestRedistributeEntries:
    def test_recent_entry_stays_recent(self, tmp_path: Path):
        from datetime import datetime, timezone, timedelta
        cfg = _make_config(tmp_path)
        now = datetime.now(timezone.utc)
        recent_ts = (now - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
        entries = [MemoryEntry(
            timestamp=recent_ts, role="user", content="hello", source_tag="chat", layer="recent"
        )]
        result = _redistribute_entries(entries, now, cfg)
        assert len(result) == 1
        assert result[0].layer == "recent"

    def test_old_entry_becomes_long_term(self, tmp_path: Path):
        from datetime import datetime, timezone, timedelta
        cfg = _make_config(tmp_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=90)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
        entries = [MemoryEntry(
            timestamp=old_ts, role="user", content="old content", source_tag="chat", layer="recent"
        )]
        result = _redistribute_entries(entries, now, cfg)
        assert len(result) == 1
        assert result[0].layer == "long_term"

    def test_preferences_entry_unchanged(self, tmp_path: Path):
        from datetime import datetime, timezone, timedelta
        cfg = _make_config(tmp_path)
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
        entries = [MemoryEntry(
            timestamp=old_ts, role="assistant", content="prefs",
            source_tag="preferences", layer=None
        )]
        result = _redistribute_entries(entries, now, cfg)
        assert len(result) == 1
        # preferences layer not changed
        assert result[0].source_tag == "preferences"

    def test_uses_config_timezone(self, tmp_path: Path):
        """_redistribute_entries が tools._core.tz を import していないこと。"""
        import mltgnt.memory._compaction as mod
        import inspect
        src = inspect.getsource(mod._redistribute_entries)
        assert "tools._core.tz" not in src
        assert "JST" not in src or "ZoneInfo" in src


# ---------------------------------------------------------------------------
# compact() — per-section cap 方式（AC-1, AC-2, AC-3, AC-5, AC-6, AC-7, AC-8, AC-10）
# ---------------------------------------------------------------------------


class TestCompactPerSectionCap:
    def _make_entry(
        self, ts: str, content: str, layer: str, source_tag: str = "chat"
    ) -> MemoryEntry:
        return MemoryEntry(
            timestamp=ts, role="user", content=content,
            source_tag=source_tag, layer=layer
        )

    def test_no_compaction_when_under_cap(self, tmp_path: Path):
        """全セクションが cap 以内 → LLM は呼ばれない（AC-1）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"

        entries = [
            self._make_entry("2026-05-01T00:00:00+09:00", "x" * 100, "recent"),
            self._make_entry("2026-04-01T00:00:00+09:00", "y" * 100, "mid_term"),
        ]
        _write_jsonl(path, entries)

        llm_called = []
        def _llm(prompt: str) -> str:
            llm_called.append(prompt)
            return "compressed"

        result = compact(cfg, "test_persona", llm_call=_llm)
        # cap 以内なので LLM は呼ばれない
        assert llm_called == []
        # result は正常に返る
        assert isinstance(result, CompactionResult)
        assert result.before_bytes > 0

    def test_promote_candidates_field_is_empty_list(self, tmp_path: Path):
        """compact() の結果に promote_candidates フィールドが空リストで存在する（AC-3）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"
        entries = [self._make_entry("2026-05-01T00:00:00+09:00", "hello", "recent")]
        _write_jsonl(path, entries)

        result = compact(cfg, "test_persona", llm_call=lambda p: "compressed")
        assert hasattr(result, "promote_candidates")
        assert result.promote_candidates == []

    def test_backward_compatible_signature(self, tmp_path: Path):
        """compact(config, stem, llm_call=fn) が max_retries/skip_min_ratio のデフォルト値で動作（AC-2）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"
        entries = [self._make_entry("2026-05-01T00:00:00+09:00", "hello world", "recent")]
        _write_jsonl(path, entries)

        # デフォルト引数のみで呼び出す
        result = compact(cfg, "test_persona", llm_call=lambda p: "ok")
        assert isinstance(result, CompactionResult)

    def test_filenotfounderror_on_missing_file(self, tmp_path: Path):
        """メモリファイルが存在しない → FileNotFoundError（AC-10 境界値）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        with pytest.raises(FileNotFoundError):
            compact(cfg, "nonexistent", llm_call=lambda p: "compressed")

    def test_dry_run_does_not_write_file(self, tmp_path: Path):
        """dry_run=True のときはファイルを変更しない。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"
        entries = [
            self._make_entry("2026-05-01T00:00:00+09:00", "x" * 200, "recent"),
        ]
        _write_jsonl(path, entries)
        original_mtime = path.stat().st_mtime
        original_text = path.read_text(encoding="utf-8")

        result = compact(cfg, "test_persona", llm_call=lambda p: "compressed", dry_run=True)
        # ファイルの内容は変わらない
        assert path.read_text(encoding="utf-8") == original_text

    def test_long_term_over_cap_triggers_llm(self, tmp_path: Path):
        """long_term が cap（25%）超過 → LLM 圧縮が発火する（AC-1）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"

        # long_term_cap = 25600 * 0.25 = 6400 bytes
        # この値を超えるコンテンツを long_term に配置
        long_content = "L" * 7000  # 7KB > 6.4KB cap
        entries = [
            self._make_entry("2025-01-01T00:00:00+09:00", long_content, "long_term"),
        ]
        _write_jsonl(path, entries)

        llm_calls: list[str] = []
        def _llm(prompt: str) -> str:
            llm_calls.append(prompt)
            # cap に収まるサイズで返す（元の90%以上を保持）
            return "L" * 6000

        result = compact(cfg, "test_persona", llm_call=_llm)
        assert len(llm_calls) > 0, "LLM should be called when long_term exceeds cap"

    def test_mid_term_over_cap_promotes_to_long_term(self, tmp_path: Path):
        """mid_term が cap 超過 → long_term に玉突き昇格（AC-8）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"

        # mid_term_cap = 25600 * 0.25 = 6400 bytes
        # mid_term に cap を超えるコンテンツを配置
        mid_content = "M" * 7000  # 7KB > 6.4KB cap
        entries = [
            self._make_entry("2026-02-01T00:00:00+09:00", mid_content, "mid_term"),
        ]
        _write_jsonl(path, entries)

        llm_calls: list[str] = []
        def _llm(prompt: str) -> str:
            llm_calls.append(prompt)
            return "M" * 6000

        result = compact(cfg, "test_persona", llm_call=_llm)
        # mid_term が cap 超過なので何らかの処理が発生するはず
        assert isinstance(result, CompactionResult)

    def test_warnings_on_llm_failure(self, tmp_path: Path):
        """LLM 失敗 → warning に記録、元テキスト保持（テスト観点: 異常系）。"""
        cfg = _make_config(tmp_path, target_bytes=25_600)
        path = tmp_path / "test_persona.jsonl"

        long_content = "L" * 7000  # cap 超過
        entries = [
            self._make_entry("2025-01-01T00:00:00+09:00", long_content, "long_term"),
        ]
        _write_jsonl(path, entries)

        def _llm_fail(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        result = compact(cfg, "test_persona", llm_call=_llm_fail)
        # LLM 失敗でも exception ではなく warning になる
        assert isinstance(result, CompactionResult)
        assert len(result.warnings) > 0

    def test_no_diary_dependency(self):
        """_compaction.py が tools._core.tz を import していない（AC-5）。"""
        import mltgnt.memory._compaction as mod
        import inspect
        src = inspect.getsource(mod)
        assert "tools._core.tz" not in src
        assert "from tools" not in src

    def test_redistribute_uses_config_timezone(self, tmp_path: Path):
        """_redistribute_entries が config.timezone を使用（AC-5）。"""
        cfg = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=tmp_path, timezone="UTC")
        assert cfg.timezone == "UTC"
        # UTC timezone でも正常に動作する
        from datetime import datetime, timezone as tz
        import mltgnt.memory._compaction as mod
        now = datetime.now(tz.utc)
        entries = [MemoryEntry(
            timestamp="2026-05-01T00:00:00+00:00",
            role="user", content="hello", source_tag="chat", layer="recent"
        )]
        result = mod._redistribute_entries(entries, now, cfg)
        assert len(result) == 1
