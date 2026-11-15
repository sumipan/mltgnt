"""
tests/test_memory_public_api.py — mltgnt.memory public API Expansion acceptance testIssue #1127 Phase B-3:
- MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text Home __all__ More- tail_utf8_bytes public alias Add"""
from __future__ import annotations



# ---------------------------------------------------------------------------
# Normal: import test
# ---------------------------------------------------------------------------

def test_import_memory_entry_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import MemoryEntry` Home ImportError without success."""
    from mltgnt.memory import MemoryEntry  # noqa: F401


def test_import_parse_jsonl_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import parse_jsonl` Home ImportError without success."""
    from mltgnt.memory import parse_jsonl  # noqa: F401


def test_import_serialize_entry_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import serialize_entry` Home ImportError without success."""
    from mltgnt.memory import serialize_entry  # noqa: F401


def test_import_assemble_entries_text_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import assemble_entries_text` Home ImportError without success."""
    from mltgnt.memory import assemble_entries_text  # noqa: F401


def test_import_tail_utf8_bytes_from_mltgnt_memory() -> None:
    """`from mltgnt.memory import tail_utf8_bytes` Home ImportError without success."""
    from mltgnt.memory import tail_utf8_bytes  # noqa: F401


def test_all_symbols_in_dunder_all() -> None:
    """MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text, tail_utf8_bytes Home __all__ Included."""
    import mltgnt.memory as m
    for symbol in ("MemoryEntry", "parse_jsonl", "serialize_entry", "assemble_entries_text", "tail_utf8_bytes"):
        assert symbol in m.__all__, f"{symbol!r} should be in mltgnt.memory.__all__"


# ---------------------------------------------------------------------------
# Normal: tail_utf8_bytes Check operation
# ---------------------------------------------------------------------------

def test_tail_utf8_bytes_equivalent_to_private() -> None:
    """`tail_utf8_bytes` Home api Inside `_tail_utf8_bytes` Returns the same results as#3023: package from private."""
    from mltgnt.memory import tail_utf8_bytes
    from mltgnt.memory.api import _tail_utf8_bytes
    assert tail_utf8_bytes("Home", 6) == _tail_utf8_bytes("Home", 6)


def test_private_tail_utf8_bytes_not_exported_from_package() -> None:
    """AC-3: `_tail_utf8_bytes` Home mltgnt.memory Home export Removed from."""
    import pytest

    import mltgnt.memory as m

    assert "_tail_utf8_bytes" not in m.__all__
    with pytest.raises(ImportError):
        from mltgnt.memory import _tail_utf8_bytes  # noqa: F401


# ---------------------------------------------------------------------------
# Abnormal/Boundary
# ---------------------------------------------------------------------------

def test_tail_utf8_bytes_empty_string() -> None:
    """`tail_utf8_bytes("", 0)` returns an empty string."""
    from mltgnt.memory import tail_utf8_bytes
    assert tail_utf8_bytes("", 0) == ""


def test_tail_utf8_bytes_max_bytes_larger_than_content() -> None:
    """`tail_utf8_bytes("abc", 100)` Home "abc" max_bytes > size)."""
    from mltgnt.memory import tail_utf8_bytes
    assert tail_utf8_bytes("abc", 100) == "abc"


def test_tail_utf8_bytes_truncates_correctly() -> None:
    """`tail_utf8_bytes` Home UTF-8 Cut the end correctly with the number of bytes."""
    from mltgnt.memory import tail_utf8_bytes
    # "a" Home UTF-8 Home 3 Byte × 5 English = 15 Byte
    # max_bytes=6 → Close 2 6 byte)= "Home"
    result = tail_utf8_bytes("Home", 6)
    assert result == "Home"


# ---------------------------------------------------------------------------
# Normal: MemoryEntry, parse_jsonl, serialize_entry, assemble_entries_text Check operation
# ---------------------------------------------------------------------------

def test_memory_entry_is_callable() -> None:
    """`mltgnt.memory.MemoryEntry` can be called as constructor."""
    from mltgnt.memory import MemoryEntry
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="test",
        source_tag="file",
    )
    assert entry.content == "test"


def test_serialize_entry_returns_string() -> None:
    """`mltgnt.memory.serialize_entry` returns a string."""
    from mltgnt.memory import MemoryEntry, serialize_entry
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="test",
        source_tag="file",
    )
    result = serialize_entry(entry)
    assert isinstance(result, str)


def test_parse_jsonl_returns_list(tmp_path) -> None:
    """`mltgnt.memory.parse_jsonl` returns a list."""
    from mltgnt.memory import MemoryEntry, serialize_entry, parse_jsonl
    path = tmp_path / "test.jsonl"
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="content",
        source_tag="file",
    )
    path.write_text(serialize_entry(entry) + "\n", encoding="utf-8")
    entries = parse_jsonl(path)
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert entries[0].content == "content"


def test_assemble_entries_text_returns_string() -> None:
    """`mltgnt.memory.assemble_entries_text` returns a string."""
    from mltgnt.memory import MemoryEntry, assemble_entries_text
    entry = MemoryEntry(
        timestamp="2030-01-01T00:00:00+09:00",
        role="user",
        content="Test",
        source_tag="file",
    )
    result = assemble_entries_text([entry])
    assert isinstance(result, str)
    assert "Test" in result
