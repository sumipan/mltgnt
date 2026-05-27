"""
mltgnt.memory.search — 関連度検索・十分性判定・反復検索。
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig
    from mltgnt.memory._scoring import ScoredEntry

from mltgnt.memory._format import MemoryEntry, assemble_entries_text, parse_jsonl
from mltgnt.memory.api import (
    _ensure_jsonl,
    _tail_utf8_bytes,
    read_memory_tail_text,
)

_log = logging.getLogger(__name__)

def _search_and_score(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_entries: int,
) -> "list[ScoredEntry]":
    """JSONL から preferences 以外のエントリをスコアリングして返す。"""
    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return []
    entries = parse_jsonl(path)
    if not entries:
        return []
    non_prefs = [e for e in entries if e.source_tag != "preferences"]
    if not non_prefs:
        return []
    # スコアリングは content テキスト単位
    entry_texts = [
        assemble_entries_text([e], preferences_heading=config.preferences_section_name).strip()
        for e in non_prefs
    ]
    from mltgnt.memory._scoring import score_entries
    scored = score_entries(query, entry_texts)
    return scored[:max_entries]


def read_memory_by_relevance(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    layers: list[str] | None = None,
) -> str:
    """クエリとの関連度が高いエントリを選択して返す。

    layers 指定時は layer がリストに含まれるエントリのみをスコアリング対象とする。
    """
    if not query:
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries, layers=layers
        )

    path = _ensure_jsonl(config, persona_stem)
    if not path.exists():
        return ""

    entries = parse_jsonl(path)
    if not entries:
        return ""

    prefs_entries = [e for e in entries if e.source_tag == "preferences"]
    non_prefs = [e for e in entries if e.source_tag != "preferences"]

    if layers is not None:
        non_prefs = [e for e in non_prefs if e.layer in layers]

    if not non_prefs:
        text = assemble_entries_text(
            prefs_entries,
            preferences_heading=config.preferences_section_name,
        )
        return _tail_utf8_bytes(text, max_bytes)

    try:
        entry_texts = [
            assemble_entries_text([e], preferences_heading=config.preferences_section_name).strip()
            for e in non_prefs
        ]
        from mltgnt.memory._scoring import score_entries
        scored = score_entries(query, entry_texts)
        top_texts = [s.text for s in scored[:max_entries]]
    except Exception as e:
        _log.warning(
            "read_memory_by_relevance: tfidf scoring error, fallback to tail text: %s", e
        )
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries, layers=layers
        )

    prefs_text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    ).rstrip("\n")
    if prefs_text and top_texts:
        text = prefs_text + "\n\n---\n\n" + "\n\n---\n\n".join(top_texts) + "\n"
    elif prefs_text:
        text = prefs_text + "\n"
    else:
        text = "\n\n---\n\n".join(top_texts) + "\n"

    return _tail_utf8_bytes(text, max_bytes)


def read_memory_with_sufficiency_check(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    llm_call: "Callable[[str], str] | None" = None,
) -> str:
    """十分性判定付き memory 検索。"""
    if llm_call is None:
        return read_memory_by_relevance(
            config, persona_stem, query, max_bytes=max_bytes, max_entries=max_entries
        )

    if not query:
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    import mltgnt.memory as _memory

    try:
        initial_scored = _memory._search_and_score(
            config, persona_stem, query, max_entries=max_entries
        )
    except Exception as e:
        _log.warning(
            "read_memory_with_sufficiency_check: initial search error, fallback: %s", e
        )
        return read_memory_tail_text(
            config, persona_stem, max_bytes=max_bytes, max_entries=max_entries
        )

    top_scored = list(initial_scored)

    if initial_scored:
        excerpt = "\n\n---\n\n".join(s.text for s in initial_scored)
        try:
            from mltgnt.memory._sufficiency import judge_sufficiency
            result = judge_sufficiency(query, excerpt, llm_call)
            if not result.sufficient and result.rewritten_query:
                try:
                    rewritten_scored = _memory._search_and_score(
                        config, persona_stem, result.rewritten_query, max_entries=max_entries
                    )
                    seen_texts: set[str] = set()
                    merged: list[ScoredEntry] = []  # type: ignore[type-arg]
                    for entry in initial_scored:
                        if entry.text not in seen_texts:
                            seen_texts.add(entry.text)
                            merged.append(entry)
                    for entry in rewritten_scored:
                        if entry.text not in seen_texts:
                            seen_texts.add(entry.text)
                            merged.append(entry)
                    merged.sort(key=lambda e: e.score, reverse=True)
                    top_scored = merged[:max_entries]
                except Exception as e2:
                    _log.warning(
                        "read_memory_with_sufficiency_check: re-search error, using initial: %s", e2
                    )
        except Exception as e:
            _log.warning(
                "read_memory_with_sufficiency_check: sufficiency check error, using initial results: %s", e
            )

    path = _ensure_jsonl(config, persona_stem)
    prefs_entries: list[MemoryEntry] = []
    if path.exists():
        all_entries = parse_jsonl(path)
        prefs_entries = [e for e in all_entries if e.source_tag == "preferences"]

    prefs_text = assemble_entries_text(
        prefs_entries,
        preferences_heading=config.preferences_section_name,
    ).rstrip("\n")
    top_texts = [s.text for s in top_scored]

    if prefs_text and top_texts:
        text = prefs_text + "\n\n---\n\n" + "\n\n---\n\n".join(top_texts) + "\n"
    elif prefs_text:
        text = prefs_text + "\n"
    elif top_texts:
        text = "\n\n---\n\n".join(top_texts) + "\n"
    else:
        return ""

    return _tail_utf8_bytes(text, max_bytes)


def read_memory_iterative(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    llm_call: "Callable[[str], str]",
    skill_paths: "list[Path] | None" = None,
    max_iterations: int = 3,
) -> str:
    """反復検索による情報収集。LLM が十分性を判定し、不足時は source 指定で再検索する。"""
    from mltgnt.memory._iterative import IterativeRetriever

    retriever = IterativeRetriever(
        config=config,
        persona_stem=persona_stem,
        skill_paths=skill_paths or [],
        llm_call=llm_call,
        max_iterations=max_iterations,
    )
    return retriever.retrieve(query, max_bytes=max_bytes, max_entries=max_entries)


def read_memory_agentic(
    config: "MemoryConfig",
    persona_stem: str,
    query: str,
    *,
    max_bytes: int,
    max_entries: int,
    llm_call: "Callable[[str], str]",
    skill_paths: "list[Path] | None" = None,
    max_iterations: int = 3,
) -> str:
    """Deprecated: use read_memory_iterative instead."""
    warnings.warn(
        "read_memory_agentic is deprecated, use read_memory_iterative",
        DeprecationWarning,
        stacklevel=2,
    )
    return read_memory_iterative(
        config,
        persona_stem,
        query,
        max_bytes=max_bytes,
        max_entries=max_entries,
        llm_call=llm_call,
        skill_paths=skill_paths,
        max_iterations=max_iterations,
    )
