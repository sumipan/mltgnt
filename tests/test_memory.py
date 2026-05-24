"""
tests/test_memory.py — mltgnt.memory のユニットテスト

設計: Issue #118 §7 AC-1, Issue #823 (JSONL 対応)
"""
from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path

import pytest

from mltgnt.config import MemoryConfig
from mltgnt.memory import (
    append_memory_entry,
    read_memory_preferences,
    read_memory_tail_text,
    memory_file_path,
    persona_memory_lock,
    compact,
    needs_compaction,
    CompactionResult,
)
from mltgnt.memory._format import (
    MemoryEntry,
    parse_jsonl,
    serialize_entry,
    assemble_entries_text,
)


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


def _ts_ago(days: float = 0, weeks: float = 0) -> str:
    """現在から指定期間前のタイムスタンプ（ISO 8601, JST）。"""
    delta = datetime.timedelta(days=days, weeks=weeks)
    dt = datetime.datetime.now(datetime.timezone.utc) - delta
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return dt.astimezone(jst).isoformat(timespec="seconds")


def _write_jsonl(path: Path, entries: list[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(serialize_entry(e) + "\n")


# ---------------------------------------------------------------------------
# JSONL ラウンドトリップ
# ---------------------------------------------------------------------------

def test_jsonl_roundtrip() -> None:
    """MemoryEntry → serialize → parse のラウンドトリップで全フィールド一致。"""
    entry = MemoryEntry(
        timestamp="2030-05-09T17:00:00+09:00",
        role="user",
        content="テスト",
        source_tag="file",
        layer=None,
        dedupe_key=None,
    )
    line = serialize_entry(entry)
    data = json.loads(line)
    assert data["timestamp"] == entry.timestamp
    assert data["role"] == entry.role
    assert data["content"] == entry.content
    assert data["source_tag"] == entry.source_tag
    assert "layer" not in data
    assert "dedupe_key" not in data


def test_jsonl_roundtrip_with_layer() -> None:
    """layer 付きエントリのラウンドトリップ。"""
    entry = MemoryEntry(
        timestamp="2030-05-09T17:00:00+09:00",
        role="assistant",
        content="絶対に再デプロイ順を逆にしない",
        source_tag="manual",
        layer="caveat",
    )
    line = serialize_entry(entry)
    data = json.loads(line)
    assert data["layer"] == "caveat"


def test_parse_jsonl_null_layer(tmp_path: Path) -> None:
    """`layer` が null の JSON 行を読み込むと MemoryEntry.layer is None。"""
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"timestamp":"2030-01-01T00:00:00+09:00","role":"user","content":"hi","source_tag":"file","layer":null}\n',
        encoding="utf-8",
    )
    entries = parse_jsonl(path)
    assert len(entries) == 1
    assert entries[0].layer is None


def test_parse_jsonl_skips_invalid_lines(tmp_path: Path) -> None:
    """不正 JSON 行はスキップされ、他エントリの読み込みに影響しない。"""
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"timestamp":"2030-01-01T00:00:00+09:00","role":"user","content":"ok","source_tag":"file"}\n'
        'INVALID JSON LINE\n'
        '{"timestamp":"2030-01-02T00:00:00+09:00","role":"user","content":"also ok","source_tag":"file"}\n',
        encoding="utf-8",
    )
    entries = parse_jsonl(path)
    assert len(entries) == 2
    assert entries[0].content == "ok"
    assert entries[1].content == "also ok"


# ---------------------------------------------------------------------------
# append_memory_entry — JSONL 形式
# ---------------------------------------------------------------------------

def test_append_memory_entry_creates_file(tmp_path: Path) -> None:
    """append_memory_entry が JSONL 形式でエントリを追記する。"""
    config = make_config(tmp_path)
    result = append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="file",
    )
    assert result is True
    mp = memory_file_path(config, "test_persona")
    assert mp.exists()
    assert mp.suffix == ".jsonl"
    entries = parse_jsonl(mp)
    assert len(entries) == 1
    assert entries[0].timestamp == "2026-04-17T10:00:00+09:00"
    assert entries[0].role == "user"
    assert entries[0].content == "hello"
    assert entries[0].source_tag == "file"


def test_append_memory_entry_with_layer(tmp_path: Path) -> None:
    """`layer="caveat"` を指定すると JSONL 行に layer フィールドが含まれる。"""
    config = make_config(tmp_path)
    append_memory_entry(
        config, "persona", "assistant", "デプロイ順を間違えないこと",
        "2030-05-09T17:00:00+09:00",
        source_tag="manual",
        layer="caveat",
    )
    mp = memory_file_path(config, "persona")
    entries = parse_jsonl(mp)
    assert entries[0].layer == "caveat"


def test_append_memory_entry_format(tmp_path: Path) -> None:
    """追記エントリが有効な JSON 1行として書き込まれる。"""
    config = make_config(tmp_path)
    append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="file",
    )
    mp = memory_file_path(config, "test_persona")
    text = mp.read_text(encoding="utf-8")
    # 1行の JSON として parse できる
    data = json.loads(text.strip())
    assert data["timestamp"] == "2026-04-17T10:00:00+09:00"
    assert data["role"] == "user"
    assert data["content"] == "hello"


# ---------------------------------------------------------------------------
# サイズガード: 破損ファイルへの追記を拒否
# ---------------------------------------------------------------------------

def test_append_rejects_corrupted_file(tmp_path: Path) -> None:
    """既存 JSONL ファイルが閾値以下の場合、追記を拒否して False を返す。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "broken")
    mp.write_text("\n", encoding="utf-8")
    result = append_memory_entry(
        config, "broken", "user", "new data", "2026-04-18T10:00:00+09:00",
        source_tag="slack",
    )
    assert result is False
    assert mp.read_text(encoding="utf-8") == "\n"


def test_append_allows_new_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = append_memory_entry(
        config, "fresh", "user", "first entry", "2026-04-18T10:00:00+09:00",
        source_tag="file",
    )
    assert result is True
    mp = memory_file_path(config, "fresh")
    assert mp.exists()
    entries = parse_jsonl(mp)
    assert entries[0].content == "first entry"


def test_append_allows_healthy_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mp = memory_file_path(config, "healthy")
    mp.write_text("{}\n" * 20, encoding="utf-8")
    result = append_memory_entry(
        config, "healthy", "user", "appended", "2026-04-18T10:00:00+09:00",
        source_tag="slack",
    )
    assert result is True
    entries = parse_jsonl(mp)
    assert any(e.content == "appended" for e in entries)


# ---------------------------------------------------------------------------
# read_memory_tail_text — JSONL + layers フィルタ
# ---------------------------------------------------------------------------

def test_read_memory_tail_text_returns_entries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for i in range(10):
        append_memory_entry(
            config, "test_persona", "user", f"entry {i}",
            f"2026-04-17T1{i:01d}:00:00+09:00",
            source_tag="file",
        )
    result = read_memory_tail_text(config, "test_persona", max_bytes=4096, max_entries=5)
    assert result
    assert "entry 5" in result or "entry 6" in result or "entry 7" in result


def test_read_memory_tail_text_nonexistent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = read_memory_tail_text(config, "nonexistent", max_bytes=1024, max_entries=5)
    assert result == ""


def test_read_memory_tail_text_max_bytes_zero(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    append_memory_entry(
        config, "test_persona", "user", "hello", "2026-04-17T10:00:00+09:00",
        source_tag="file",
    )
    result = read_memory_tail_text(config, "test_persona", max_bytes=0, max_entries=5)
    assert result == ""


def test_read_memory_tail_text_layers_filter(tmp_path: Path) -> None:
    """`layers=["caveat"]` 指定時、caveat のエントリのみ返す。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "persona")
    entries = [
        MemoryEntry("2030-01-01T00:00:00+09:00", "user", "通常エントリ", "file"),
        MemoryEntry("2030-01-02T00:00:00+09:00", "assistant", "caveatエントリ", "manual", layer="caveat"),
        MemoryEntry("2030-01-03T00:00:00+09:00", "user", "通常エントリ2", "file"),
    ]
    _write_jsonl(mp, entries)

    result = read_memory_tail_text(config, "persona", max_bytes=4096, max_entries=10, layers=["caveat"])
    assert "caveatエントリ" in result
    assert "通常エントリ" not in result


def test_read_memory_tail_text_layers_none_returns_all(tmp_path: Path) -> None:
    """`layers=None` は全エントリを返す（デフォルト動作）。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "persona")
    entries = [
        MemoryEntry("2030-01-01T00:00:00+09:00", "user", "通常", "file"),
        MemoryEntry("2030-01-02T00:00:00+09:00", "assistant", "caveat", "manual", layer="caveat"),
    ]
    _write_jsonl(mp, entries)

    result = read_memory_tail_text(config, "persona", max_bytes=4096, max_entries=10, layers=None)
    assert "通常" in result
    assert "caveat" in result


# ---------------------------------------------------------------------------
# read_memory_preferences
# ---------------------------------------------------------------------------

def test_read_memory_preferences_extracts_section(tmp_path: Path) -> None:
    """source_tag="preferences" エントリを `## ユーザーの好み・傾向` 見出しで返す。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test_persona")
    _write_jsonl(mp, [
        MemoryEntry("1970-01-01T00:00:00+00:00", "system", "好みのテキストです。", "preferences"),
        MemoryEntry("2026-04-17T10:00:00+09:00", "user", "エントリです。", "file"),
    ])
    result = read_memory_preferences(config, "test_persona")
    assert "ユーザーの好み・傾向" in result
    assert "好みのテキストです。" in result


def test_read_memory_preferences_nonexistent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    result = read_memory_preferences(config, "nonexistent")
    assert result == ""


def test_read_memory_preferences_no_section(tmp_path: Path) -> None:
    """preferences エントリがないファイルは空文字列を返す。"""
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test_persona")
    _write_jsonl(mp, [
        MemoryEntry("2026-04-17T10:00:00+09:00", "user", "エントリ", "file"),
    ])
    result = read_memory_preferences(config, "test_persona")
    assert result == ""


# ---------------------------------------------------------------------------
# 重複排除
# ---------------------------------------------------------------------------

def test_dedupe_key_prevents_second_write(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    r1 = append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17T10:00:00+09:00",
        source_tag="file", dedupe_key="key1",
    )
    r2 = append_memory_entry(
        config, "test_persona", "user", "hi", "2026-04-17T10:00:00+09:00",
        source_tag="file", dedupe_key="key1",
    )
    assert r1 is True
    assert r2 is True
    entries = parse_jsonl(memory_file_path(config, "test_persona"))
    assert len(entries) == 1


def test_no_dedupe_key_allows_duplicate(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    for _ in range(2):
        append_memory_entry(
            config, "test_persona", "user", "hi", "2026-04-17T10:00:00+09:00",
            source_tag="file", dedupe_key=None,
        )
    entries = parse_jsonl(memory_file_path(config, "test_persona"))
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# ロック
# ---------------------------------------------------------------------------

def test_persona_memory_lock_blocks_second_acquire(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    ready = threading.Event()

    def hold():
        with persona_memory_lock(config, "lock_test", timeout_sec=5.0) as ok:
            assert ok
            ready.set()
            time.sleep(0.5)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    assert ready.wait(timeout=2.0)
    with persona_memory_lock(config, "lock_test", timeout_sec=0.05) as ok2:
        assert not ok2
    t.join(timeout=3.0)


# ---------------------------------------------------------------------------
# memory_file_path
# ---------------------------------------------------------------------------

def test_memory_file_path(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path = memory_file_path(config, "test_persona")
    assert path == config.chat_memory_dir / "test_persona.jsonl"


def test_memory_config_default_chat_memory_dir(tmp_path: Path) -> None:
    config = MemoryConfig(chat_dir=tmp_path)
    path = memory_file_path(config, "persona")
    assert path == tmp_path / "memory" / "persona.jsonl"


def test_memory_config_explicit_chat_memory_dir(tmp_path: Path) -> None:
    other = tmp_path / "other"
    config = MemoryConfig(chat_dir=tmp_path, chat_memory_dir=other)
    path = memory_file_path(config, "persona")
    assert path == other / "persona.jsonl"


def test_memory_config_new_fields() -> None:
    config = MemoryConfig(chat_dir=Path("/tmp/chat"))
    assert config.raw_days == 7
    assert config.mid_weeks == 3
    assert config.compact_threshold_bytes == 40_960
    assert config.compact_target_bytes == 25_600
    assert config.preferences_section_name == "ユーザーの好み・傾向"
    assert config.protected_layers == ("caveat",)


def test_memory_config_new_fields_custom() -> None:
    config = MemoryConfig(
        chat_dir=Path("/tmp/chat"),
        raw_days=14,
        mid_weeks=4,
        compact_threshold_bytes=81_920,
        compact_target_bytes=51_200,
        preferences_section_name="User Preferences",
        protected_layers=(),
    )
    assert config.raw_days == 14
    assert config.mid_weeks == 4
    assert config.compact_threshold_bytes == 81_920
    assert config.compact_target_bytes == 51_200
    assert config.preferences_section_name == "User Preferences"
    assert config.protected_layers == ()


# ---------------------------------------------------------------------------
# 旧 API パブリックインタフェース除去の確認
# ---------------------------------------------------------------------------

def test_old_api_symbols_not_in_all() -> None:
    """MemorySections, parse_memory, format_memory, assemble_memory, migrate_markdown_to_jsonl が __all__ に含まれない。"""
    import mltgnt.memory._format as fmt
    for symbol in ("MemorySections", "parse_memory", "format_memory", "assemble_memory", "migrate_markdown_to_jsonl"):
        assert symbol not in fmt.__all__, f"{symbol} should not be in __all__"


def test_ensure_jsonl_no_md_fallback(tmp_path: Path) -> None:
    """.jsonl なし・.md ありの場合、自動マイグレーションせず .jsonl パスをそのまま返す（空文字列）。"""
    config = make_config(tmp_path)
    md_path = memory_file_path(config, "persona").with_suffix(".md")
    md_path.write_text("## ユーザーの好み・傾向\n\n好みの内容\n", encoding="utf-8")

    result = read_memory_tail_text(config, "persona", max_bytes=4096, max_entries=20)
    assert result == ""
    assert not memory_file_path(config, "persona").exists()


def test_old_api_functions_not_accessible() -> None:
    """parse_memory, format_memory, assemble_memory, MemorySections がモジュール属性として存在しない。"""
    import mltgnt.memory._format as fmt
    for symbol in ("parse_memory", "format_memory", "assemble_memory", "MemorySections"):
        assert not hasattr(fmt, symbol), f"{symbol} should not be a public attribute"


# ---------------------------------------------------------------------------
# assemble_entries_text
# ---------------------------------------------------------------------------

def test_assemble_entries_text_preferences_heading() -> None:
    """preferences エントリは '## ユーザーの好み・傾向' 見出しで出力される。"""
    entries = [
        MemoryEntry("1970-01-01T00:00:00+00:00", "system", "好みの内容", "preferences"),
    ]
    result = assemble_entries_text(entries)
    assert "## ユーザーの好み・傾向" in result
    assert "好みの内容" in result


def test_assemble_entries_text_regular_entry() -> None:
    """通常エントリは `## timestamp — role` 形式で出力される。"""
    entries = [
        MemoryEntry("2030-05-09T17:00:00+09:00", "user", "コンテンツ", "file"),
    ]
    result = assemble_entries_text(entries)
    assert "## 2030-05-09T17:00:00+09:00 — user" in result
    assert "コンテンツ" in result


def test_assemble_entries_text_separator() -> None:
    """複数エントリが --- で区切られる。"""
    entries = [
        MemoryEntry("2030-01-01T00:00:00+09:00", "user", "A", "file"),
        MemoryEntry("2030-01-02T00:00:00+09:00", "user", "B", "file"),
    ]
    result = assemble_entries_text(entries)
    assert "---" in result



# ---------------------------------------------------------------------------
# needs_compaction
# ---------------------------------------------------------------------------

def test_needs_compaction_false_when_file_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert needs_compaction(config, "nonexistent") is False


def test_needs_compaction_false_when_below_threshold(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    mp = memory_file_path(config, "test")
    mp.write_text('{"timestamp":"t","role":"u","content":"x","source_tag":"file"}\n', encoding="utf-8")
    assert needs_compaction(config, "test") is False


def test_needs_compaction_true_when_above_threshold(tmp_path: Path) -> None:
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
        compact_threshold_bytes=100,
    )
    mp = memory_file_path(config, "test")
    mp.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        '{"timestamp":"t","role":"u","content":"x","source_tag":"file"}'
        for _ in range(10)
    ) + "\n"
    mp.write_text(content, encoding="utf-8")
    assert needs_compaction(config, "test") is True


# ---------------------------------------------------------------------------
# compact — JSONL ベース + protected_layers
# ---------------------------------------------------------------------------


def _make_compact_entries(
    tmp_path: Path,
    *,
    compact_threshold: int = 10,
) -> tuple[MemoryConfig, str]:
    """compact テスト用の JSONL ファイルを作成して (config, persona) を返す。

    per-section cap 方式（Issue #1135）に合わせ、長期セクションが cap を超えるサイズの
    エントリを生成する。compact_target_bytes=4096、long_term_cap=1024（25%）。
    """
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
        compact_threshold_bytes=compact_threshold,
        compact_target_bytes=4_096,  # 小さな target: cap = 4096 * 0.25 = 1024 bytes
        raw_days=7,
        mid_weeks=3,
    )
    (tmp_path / "memory").mkdir(exist_ok=True)
    mp = memory_file_path(config, "persona")

    entries = [
        MemoryEntry("1970-01-01T00:00:00+00:00", "system", "好みの内容", "preferences"),
        # 長期エントリ (> 3週間前) — cap(1024B) を超えるサイズ
        MemoryEntry(_ts_ago(days=60), "user", "長期ログの内容 " + "x" * 1100, "file", layer="long_term"),
        # 中期エントリ (7日〜3週間前) — cap(1024B) を超えるサイズ
        MemoryEntry(_ts_ago(days=14), "user", "中期ログの内容 " + "y" * 1100, "file", layer="mid_term"),
        # 直近エントリ (7日以内)
        MemoryEntry(_ts_ago(days=2), "user", "直近ログの内容", "file", layer="recent"),
    ]
    _write_jsonl(mp, entries)
    return config, "persona"


def _identity_llm(prompt: str) -> str:
    parts = prompt.split("\n\n", 1)
    return parts[-1] if len(parts) > 1 else prompt


def test_compact_calls_llm_per_group_and_writes_result(tmp_path: Path) -> None:
    """compact が cap 超過のセクションに対して llm_call を呼び出し、結果を JSONL に書き込む。

    per-section cap 方式（Issue #1135）: long_term / mid_term が cap 超過の場合に LLM が呼ばれる。
    """
    config, persona = _make_compact_entries(tmp_path)

    calls: list[str] = []

    def mock_llm(prompt: str) -> str:
        calls.append(prompt)
        # cap 内に収まるコンパクト結果を返す（90% 程度のサイズ）
        return "（コンパクト）圧縮済み内容 " + "z" * 800

    result = compact(config, persona, llm_call=mock_llm)

    # per-section cap 超過セクションに対して LLM が呼ばれる（1回以上）
    assert len(calls) >= 1
    assert isinstance(result, CompactionResult)
    assert result.before_bytes > 0
    assert result.after_bytes > 0

    entries = parse_jsonl(memory_file_path(config, "persona"))
    contents = [e.content for e in entries]
    assert any("好みの内容" in c for c in contents)
    assert any("直近ログの内容" in c for c in contents)


def test_compact_dry_run_does_not_write(tmp_path: Path) -> None:
    config, persona = _make_compact_entries(tmp_path)
    original = memory_file_path(config, persona).read_text(encoding="utf-8")

    compact(config, persona, llm_call=_identity_llm, dry_run=True)

    assert memory_file_path(config, persona).read_text(encoding="utf-8") == original


def test_compact_all_llm_failures_fallback(tmp_path: Path) -> None:
    """全 LLM 呼び出しが失敗しても元テキストにフォールバックして書き込みが行われる。

    per-section cap 方式（Issue #1135）: cap 超過セクションに対して LLM が試みられ、
    失敗時は元テキストを保持して warning に記録する。
    """
    config, persona = _make_compact_entries(tmp_path)
    call_count = 0

    def failing_llm(_: str) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("LLM unavailable")

    result = compact(config, persona, llm_call=failing_llm)

    # cap 超過セクションに対して LLM が試みられる（1回以上）
    assert call_count >= 1
    assert len(result.warnings) >= 1
    assert result.after_bytes > 0


def test_compact_raises_file_not_found(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(FileNotFoundError):
        compact(config, "nonexistent", llm_call=lambda p: p)


def test_compact_preserves_preferences(tmp_path: Path) -> None:
    """compact 後も preferences エントリが保持される。"""
    config, persona = _make_compact_entries(tmp_path)
    compact(config, persona, llm_call=_identity_llm)

    entries = parse_jsonl(memory_file_path(config, persona))
    prefs = [e for e in entries if e.source_tag == "preferences"]
    assert prefs
    assert "好みの内容" in prefs[0].content


def test_compact_preserves_protected_layers(tmp_path: Path) -> None:
    """compact 後、`layer="caveat"` のエントリが JSONL に残存する。"""
    config, persona = _make_compact_entries(tmp_path)
    mp = memory_file_path(config, persona)

    # caveat エントリを追加
    caveat_entry = MemoryEntry(
        _ts_ago(days=90),
        "assistant",
        "絶対に再デプロイ順を逆にしない",
        "manual",
        layer="caveat",
    )
    existing = parse_jsonl(mp)
    _write_jsonl(mp, existing + [caveat_entry])

    compact(config, persona, llm_call=_identity_llm)

    entries = parse_jsonl(mp)
    caveat_entries = [e for e in entries if e.layer == "caveat"]
    assert caveat_entries
    assert "絶対に再デプロイ順を逆にしない" in caveat_entries[0].content


def test_compact_protected_layer_content_unchanged(tmp_path: Path) -> None:
    """compact 後、caveat エントリの content が変更されていない。"""
    config, persona = _make_compact_entries(tmp_path)
    mp = memory_file_path(config, persona)
    original_caveat_content = "この内容は変更されてはならない"
    caveat_entry = MemoryEntry(
        _ts_ago(days=90),
        "assistant",
        original_caveat_content,
        "manual",
        layer="caveat",
    )
    existing = parse_jsonl(mp)
    _write_jsonl(mp, existing + [caveat_entry])

    compact(config, persona, llm_call=_identity_llm)

    entries = parse_jsonl(mp)
    caveat_entries = [e for e in entries if e.layer == "caveat"]
    assert caveat_entries[0].content == original_caveat_content


def test_compact_no_protected_layers_compacts_all(tmp_path: Path) -> None:
    """`protected_layers=()` のとき caveat エントリもコンパクション対象になる。

    per-section cap 方式（Issue #1135）: protected_layers=() の場合、caveat エントリも
    通常の layer 分類（long_term/mid_term/recent）に従い、cap 超過時に LLM 圧縮対象となる。
    """
    config = MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=tmp_path / "memory",
        compact_threshold_bytes=10,
        compact_target_bytes=4_096,  # cap = 1024 bytes per section
        raw_days=7,
        mid_weeks=3,
        protected_layers=(),
    )
    (tmp_path / "memory").mkdir(exist_ok=True)
    mp = memory_file_path(config, "persona")
    entries = [
        # caveat エントリ（long_term layer、cap 超過サイズ）
        MemoryEntry(_ts_ago(days=90), "assistant", "caveat内容 " + "c" * 1100, "manual", layer="long_term"),
        MemoryEntry(_ts_ago(days=2), "user", "直近内容", "file", layer="recent"),
    ]
    _write_jsonl(mp, entries)

    calls: list[str] = []

    def track_llm(prompt: str) -> str:
        calls.append(prompt)
        # max_ratio(0.90) ガードを通過する圧縮結果を返す（existing の 95% 程度）
        # existing は ~1110 bytes → 結果は ~1050 bytes 以上必要
        return "圧縮済みデータ " + "z" * 1080

    compact(config, "persona", llm_call=track_llm)

    # cap 超過の long_term に対して LLM が呼ばれる
    assert len(calls) >= 1
    result_entries = parse_jsonl(mp)
    # LLM 圧縮が実行されたので元のロングコンテンツは残らない
    original_content_entries = [e for e in result_entries if "c" * 100 in e.content]
    assert len(original_content_entries) == 0  # LLM が圧縮したので元の "ccc..." コンテンツはない



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# extract_and_append 削除確認 (Issue #915)
# ---------------------------------------------------------------------------

def test_extract_and_append_not_in_all() -> None:
    """extract_and_append が __all__ に含まれていないことを確認。"""
    import mltgnt.memory as m
    assert "extract_and_append" not in m.__all__


def test_extract_and_append_not_importable() -> None:
    """`from mltgnt.memory import extract_and_append` が ImportError を返す。"""
    with pytest.raises(ImportError):
        from mltgnt.memory import extract_and_append  # noqa: F401
