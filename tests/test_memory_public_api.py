"""
tests/test_memory_public_api.py — mltgnt.memory public API 拡張の受け入れテスト

Issue #1127 Phase B-3:
- MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text を __all__ に追加
- tail_utf8_bytes public alias を追加
"""
from __future__ import annotations



# ---------------------------------------------------------------------------
# 正常系: import テスト
# ---------------------------------------------------------------------------

def test_import_memory_entry_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import MemoryEntry` が ImportError なしで成功する。"""
    from mltgnt.memory import MemoryEntry  # noqa: F401


def test_import_parse_jsonl_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import parse_jsonl` が ImportError なしで成功する。"""
    from mltgnt.memory import parse_jsonl  # noqa: F401


def test_import_serialize_entry_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import serialize_entry` が ImportError なしで成功する。"""
    from mltgnt.memory import serialize_entry  # noqa: F401


def test_import_assemble_entries_text_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import assemble_entries_text` が ImportError なしで成功する。"""
    from mltgnt.memory import assemble_entries_text  # noqa: F401


def test_import_tail_utf8_bytes_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import tail_utf8_bytes` が ImportError なしで成功する。"""
    from mltgnt.memory import tail_utf8_bytes  # noqa: F401


def test_all_symbols_in_dunder_all() -> None:
    """MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text, tail_utf8_bytes が __all__ に含まれる。"""
    import mltgnt.memory as m
    for symbol in ("MemoryEntry", "parse_jsonl", "serialize_entry", "assemble_entries_text", "tail_utf8_bytes"):
        assert symbol in m.__all__, f"{symbol!r} should be in mltgnt.memory.__all__"


# ---------------------------------------------------------------------------
# 正常系: tail_utf8_bytes の動作確認
# ---------------------------------------------------------------------------

def test_tail_utf8_bytes_equivalent_to_private() -> None:
    """`tail_utf8_bytes("あいうえお", 6)` が `_tail_utf8_bytes("あいうえお", 6)` と同じ結果を返す。"""
    from mltgnt.memory import tail_utf8_bytes
    from mltgnt.memory import _tail_utf8_bytes
    assert tail_utf8_bytes("あいうえお", 6) == _tail_utf8_bytes("あいうえお", 6)


# ---------------------------------------------------------------------------
# 異常系・境界値
# ---------------------------------------------------------------------------

def test_tail_utf8_bytes_empty_string() -> None:
    """`tail_utf8_bytes("", 0)` が空文字列を返す。"""
    from mltgnt.memory import tail_utf8_bytes
    assert tail_utf8_bytes("", 0) == ""


def test_tail_utf8_bytes_max_bytes_larger_than_content() -> None:
    """`tail_utf8_bytes("abc", 100)` が "abc" をそのまま返す（max_bytes > 実サイズ）。"""
    from mltgnt.memory import tail_utf8_bytes
    assert tail_utf8_bytes("abc", 100) == "abc"


def test_tail_utf8_bytes_truncates_correctly() -> None:
    """`tail_utf8_bytes` が UTF-8 バイト数で正しく末尾を切り出す。"""
    from mltgnt.memory import tail_utf8_bytes
    # "あ" は UTF-8 で 3 バイト × 5 文字 = 15 バイト
    # max_bytes=6 → 末尾 2 文字分（6 バイト）= "えお"
    result = tail_utf8_bytes("あいうえお", 6)
    assert result == "えお"


# ---------------------------------------------------------------------------
# 正常系: MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text の動作確認
# ---------------------------------------------------------------------------

def test_memory_entry_is_callable() -> None:
    """`mltgnt.memory.MemoryEntry` はコンストラクタとして呼び出せる。"""
    from mltgnt.memory import MemoryEntry
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="テスト",
        source_tag="file",
    )
    assert entry.content == "テスト"


def test_serialize_entry_returns_string() -> None:
    """`mltgnt.memory.serialize_entry` が文字列を返す。"""
    from mltgnt.memory import MemoryEntry, serialize_entry
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="テスト",
        source_tag="file",
    )
    result = serialize_entry(entry)
    assert isinstance(result, str)


def test_parse_jsonl_returns_list(tmp_path) -> None:
    """`mltgnt.memory.parse_jsonl` がリストを返す。"""
    from mltgnt.memory import MemoryEntry, serialize_entry, parse_jsonl
    path = tmp_path / "test.jsonl"
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="内容",
        source_tag="file",
    )
    path.write_text(serialize_entry(entry) + "\n", encoding="utf-8")
    entries = parse_jsonl(path)
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0].content == "内容"


def test_assemble_entries_text_returns_string() -> None:
    """`mltgnt.memory.assemble_entries_text` が文字列を返す。"""
    from mltgnt.memory import MemoryEntry, assemble_entries_text
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="テスト内容",
        source_tag="file",
    )
    result = assemble_entries_text([entry])
    assert isinstance(result, str)
    assert "テスト内容" in result
