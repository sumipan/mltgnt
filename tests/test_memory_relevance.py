"""
tests/test_memory_relevance.py — integration tests for read_memory_by_relevance()

TC1: score-ordered selection
TC2: preferences always included
TC3: max_entries limit
TC4: max_bytes limit
TC5: Japanese text support
TC6: scoring-error fallback
TC7: empty memory
TC8: preferences only
TC9: empty query
TC10: single entry

Note: Issue #198 switched from embedding-based to TF-IDF-based scoring.
The embedding_call parameter was removed; TF-IDF is used locally.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from mltgnt.config import MemoryConfig
from mltgnt.memory import read_memory_by_relevance, read_memory_tail_text, memory_file_path, read_memory_with_sufficiency_check
from mltgnt.memory._scoring import ScoredEntry


def make_config(tmp_path: Path) -> MemoryConfig:
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    return MemoryConfig(
        chat_dir=tmp_path,
        chat_memory_dir=mem_dir,
    )


def _write_memory(config: MemoryConfig, persona: str, content: str) -> None:
    # Write JSONL to the .jsonl path.
    jsonl_path = memory_file_path(config, persona)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# TC1: score-ordered selection
# ---------------------------------------------------------------------------


MEMORY_THREE_ENTRIES = (
    '{"timestamp":"2026-01-01T10:00:00+09:00","role":"user","content":"Discussed recipes and learned to make pasta.","source_tag":"file"}\n'
    '{"timestamp":"2026-01-02T10:00:00+09:00","role":"user","content":"Python  decorators were studied. Code reuse improved.","source_tag":"file"}\n'
    '{"timestamp":"2026-01-03T10:00:00+09:00","role":"user","content":"The weather was sunny and warmer today.","source_tag":"file"}\n'
)


def test_tc1_score_ordering(tmp_path: Path) -> None:
    """TC1: for a Python query, the programming entry ranks first."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python decorator",
        max_bytes=4096,
        max_entries=3,
    )

    # Programming entry must appear before cooking/weather
    prog_idx = result.find("Python  decorators")
    cook_idx = result.find("recipes")
    assert prog_idx != -1
    assert cook_idx != -1
    assert prog_idx < cook_idx


# ---------------------------------------------------------------------------
# TC2: preferences always included
# ---------------------------------------------------------------------------


MEMORY_WITH_PREFERENCES = (
    '{"timestamp":"1970-01-01T00:00:00+00:00","role":"system","content":"Good at programming.Python  is mainly used.","source_tag":"preferences"}\n'
    '{"timestamp":"2026-01-01T10:00:00+09:00","role":"user","content":"Discussed recipes.","source_tag":"file"}\n'
    '{"timestamp":"2026-01-02T10:00:00+09:00","role":"user","content":"Discussed the weather.","source_tag":"file"}\n'
)


def test_tc2_preferences_always_included(tmp_path: Path) -> None:
    """TC2: preferences section is included regardless of scoring."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_WITH_PREFERENCES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "today weather",
        max_bytes=4096,
        max_entries=1,
    )

    assert config.preferences_section_name in result
    assert "Python  is mainly used" in result


# ---------------------------------------------------------------------------
# TC3: max_entries limit
# ---------------------------------------------------------------------------


def _make_10_entries_memory() -> str:
    import json
    lines = []
    for i in range(10):
        lines.append(json.dumps({
            "timestamp": f"2026-01-{i+1:02d}T10:00:00+09:00",
            "role": "user",
            "content": f"entry {i}",
            "source_tag": "file",
        }, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def test_tc3_max_entries_limit(tmp_path: Path) -> None:
    """TC3: 10 entries with max_entries=3 → at most 3 returned."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", _make_10_entries_memory())

    result = read_memory_by_relevance(
        config,
        "persona",
        "entry query",
        max_bytes=65536,
        max_entries=3,
    )

    # Occurrences of "entry N" ≤ 3
    entry_count = sum(1 for i in range(10) if f"entry {i}" in result)
    assert entry_count <= 3


# ---------------------------------------------------------------------------
# TC4: max_bytes limit
# ---------------------------------------------------------------------------


def test_tc4_max_bytes_limit(tmp_path: Path) -> None:
    """TC4: when top entries exceed max_bytes, output stays within the byte budget."""
    import json
    config = make_config(tmp_path)
    lines = [
        json.dumps({
            "timestamp": f"2026-01-{i+1:02d}T10:00:00+09:00",
            "role": "user",
            "content": "x" * 500,
            "source_tag": "file",
        }, ensure_ascii=False)
        for i in range(10)
    ]
    big_entries = "\n".join(lines) + "\n"
    _write_memory(config, "persona", big_entries)

    max_bytes = 800
    result = read_memory_by_relevance(
        config,
        "persona",
        "test",
        max_bytes=max_bytes,
        max_entries=10,
    )

    assert len(result.encode("utf-8")) <= max_bytes


# ---------------------------------------------------------------------------
# TC5: Japanese text support
# ---------------------------------------------------------------------------


def test_tc5_japanese_text(tmp_path: Path) -> None:
    """TC5: TF-IDF vectorization works on Japanese text and returns scores."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python decorator code",
        max_bytes=4096,
        max_entries=3,
    )

    # Result must be non-empty and include an entry
    assert result
    assert "Python  decorators" in result


# ---------------------------------------------------------------------------
# TC6: scoring-error fallback
# ---------------------------------------------------------------------------


def test_tc6_scoring_error_fallback(tmp_path: Path, caplog) -> None:
    """TC6: if score_entries() raises, return the same as read_memory_tail_text()."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_tail_text(
        config, "persona", max_bytes=4096, max_entries=5
    )

    with caplog.at_level(logging.WARNING, logger="mltgnt.memory"):
        with patch("mltgnt.memory._scoring.score_entries", side_effect=RuntimeError("TF-IDF error")):
            result = read_memory_by_relevance(
                config,
                "persona",
                "something",
                max_bytes=4096,
                max_entries=5,
            )

    assert result == expected
    assert any(
        "tfidf" in r.message.lower()
        or "error" in r.message.lower()
        or "fallback" in r.message.lower()
        or "scoring" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# TC7: empty memory
# ---------------------------------------------------------------------------


def test_tc7_empty_memory(tmp_path: Path) -> None:
    """TC7: missing memory file returns an empty string."""
    config = make_config(tmp_path)

    result = read_memory_by_relevance(
        config,
        "nonexistent",
        "some question",
        max_bytes=4096,
        max_entries=5,
    )

    assert result == ""


# ---------------------------------------------------------------------------
# TC8: preferences only
# ---------------------------------------------------------------------------


MEMORY_PREFERENCES_ONLY = (
    '{"timestamp":"1970-01-01T00:00:00+00:00","role":"system","content":"Likes programming.","source_tag":"preferences"}\n'
)


def test_tc8_preferences_only(tmp_path: Path) -> None:
    """TC8: preferences-only memory → preferences-only output."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_PREFERENCES_ONLY)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python  about ",
        max_bytes=4096,
        max_entries=5,
    )

    assert config.preferences_section_name in result
    assert "likes programming" in result.lower()


# ---------------------------------------------------------------------------
# TC9: empty query
# ---------------------------------------------------------------------------


def test_tc9_empty_query_fallback(tmp_path: Path) -> None:
    """TC9: empty query falls back to read_memory_tail_text()."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_tail_text(
        config, "persona", max_bytes=4096, max_entries=5
    )

    result = read_memory_by_relevance(
        config,
        "persona",
        "",
        max_bytes=4096,
        max_entries=5,
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC10: single entry
# ---------------------------------------------------------------------------


MEMORY_SINGLE_ENTRY = (
    '{"timestamp":"2026-01-01T10:00:00+09:00","role":"user","content":"Python  decorators were studied.","source_tag":"file"}\n'
)


def test_tc10_single_entry(tmp_path: Path) -> None:
    """TC10: TF-IDF works with a single entry."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_SINGLE_ENTRY)

    result = read_memory_by_relevance(
        config,
        "persona",
        "Python",
        max_bytes=4096,
        max_entries=5,
    )

    assert "Python  decorators" in result


# ---------------------------------------------------------------------------
# Phase 2 tests — read_memory_with_sufficiency_check()
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# TC1: llm_call returns SUFFICIENT → same result as read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_suf_tc1_sufficient_same_as_relevance(tmp_path: Path) -> None:
    """TC1: llm_call returns SUFFICIENT → same as read_memory_by_relevance."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_by_relevance(
        config, "persona", "Python decorator", max_bytes=4096, max_entries=3
    )

    result = read_memory_with_sufficiency_check(
        config,
        "persona",
        "Python decorator",
        max_bytes=4096,
        max_entries=3,
        llm_call=lambda p: "SUFFICIENT",
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC2: llm_call returns INSUFFICIENT → re-search, results merged
# ---------------------------------------------------------------------------


def test_suf_tc2_insufficient_merges_results(tmp_path: Path) -> None:
    """TC2: INSUFFICIENT → re-search; all three entries appear in the result."""
    config = make_config(tmp_path)
    # Create the memory file (needed for preferences loading)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    entry_a = ScoredEntry("entry-A: project progress", 0.9)
    entry_b = ScoredEntry("entry-B: DB connection settings", 0.8)
    entry_c = ScoredEntry("entry-C: cooking", 0.3)

    call_count = [0]

    def mock_search(cfg, persona, q, *, max_entries):
        call_count[0] += 1
        if call_count[0] == 1:
            return [entry_a, entry_c]
        else:
            return [entry_b, entry_a]

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "project",
            max_bytes=4096,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\nDB connection details",
        )

    assert "entry-A" in result
    assert "entry-B" in result
    assert "entry-C" in result
    assert call_count[0] == 2


# ---------------------------------------------------------------------------
# TC3: duplicate deduplication
# ---------------------------------------------------------------------------


def test_suf_tc3_deduplication(tmp_path: Path) -> None:
    """TC3: first and re-search return the same entries → no duplicates."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    entries = [
        ScoredEntry("entry-A", 0.9),
        ScoredEntry("entry-B", 0.8),
    ]

    with patch("mltgnt.memory._search_and_score", return_value=entries):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "test",
            max_bytes=4096,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\nextra query",
        )

    # entry-A must not appear more than once
    assert result.count("entry-A") == 1
    assert result.count("entry-B") == 1


# ---------------------------------------------------------------------------
# TC4: max_entries limit after merge
# ---------------------------------------------------------------------------


def test_suf_tc4_max_entries_after_merge(tmp_path: Path) -> None:
    """TC4: after merge, result still respects max_entries."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    first_entries = [ScoredEntry(f"entry-{i}", float(10 - i) / 10) for i in range(8)]
    second_entries = [ScoredEntry(f"entry-{i}", float(10 - i) / 10) for i in range(4, 10)]

    call_count = [0]

    def mock_search(cfg, persona, q, *, max_entries):
        call_count[0] += 1
        if call_count[0] == 1:
            return first_entries
        else:
            return second_entries

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        result = read_memory_with_sufficiency_check(
            config,
            "persona",
            "test",
            max_bytes=65536,
            max_entries=10,
            llm_call=lambda p: "INSUFFICIENT\nMEMORY\nextra query",
        )

    # Count entries present in the result (entry-0 … entry-9)
    entry_count = sum(1 for i in range(10) if f"entry-{i}" in result)
    assert entry_count <= 10


# ---------------------------------------------------------------------------
# TC5: llm_call=None → same as read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_suf_tc5_no_llm_call_same_as_relevance(tmp_path: Path) -> None:
    """TC5: llm_call=None → same as read_memory_by_relevance()."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    expected = read_memory_by_relevance(
        config, "persona", "Python", max_bytes=4096, max_entries=3
    )

    result = read_memory_with_sufficiency_check(
        config,
        "persona",
        "Python",
        max_bytes=4096,
        max_entries=3,
        llm_call=None,
    )

    assert result == expected


# ---------------------------------------------------------------------------
# TC7 integration: llm_call raises → initial result returned with warning
# ---------------------------------------------------------------------------


def test_suf_tc7_llm_raises_returns_initial(tmp_path: Path, caplog) -> None:
    """TC7 integration: judge_sufficiency raises → return initial result and warn."""
    config = make_config(tmp_path)
    _write_memory(config, "persona", MEMORY_THREE_ENTRIES)

    initial_entries = [ScoredEntry("Python  decorators were studied. Code reuse improved.", 0.9)]

    def mock_search(cfg, persona, q, *, max_entries):
        return initial_entries

    with patch("mltgnt.memory._search_and_score", side_effect=mock_search):
        with patch("mltgnt.memory._sufficiency.judge_sufficiency", side_effect=RuntimeError("LLM error")):
            with caplog.at_level(logging.WARNING, logger="mltgnt.memory"):
                result = read_memory_with_sufficiency_check(
                    config,
                    "persona",
                    "Python",
                    max_bytes=4096,
                    max_entries=5,
                    llm_call=lambda p: "SUFFICIENT",
                )

    assert "Python  decorators" in result
    assert any(
        "sufficiency" in r.message.lower() or "error" in r.message.lower()
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# layers filter: read_memory_by_relevance
# ---------------------------------------------------------------------------


def test_read_memory_by_relevance_layers_filter(tmp_path: Path) -> None:
    """With layers=["learning"], only layer="learning" entries are returned."""
    from mltgnt.memory._format import MemoryEntry, serialize_entry

    config = make_config(tmp_path)
    mp = memory_file_path(config, "persona")
    mp.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        MemoryEntry("2030-01-01T00:00:00+09:00", "user", "learning entry", "file", layer="learning"),
        MemoryEntry("2030-01-02T00:00:00+09:00", "user", "caveat entry", "file", layer="caveat"),
        MemoryEntry("2030-01-03T00:00:00+09:00", "user", "normal entry", "file"),
    ]
    with mp.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(serialize_entry(e) + "\n")

    result = read_memory_by_relevance(
        config, "persona", "learning",
        max_bytes=4096, max_entries=10, layers=["learning"],
    )
    assert "learning entry" in result
    assert "caveat entry" not in result
    assert "normal entry" not in result
