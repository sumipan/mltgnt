"""
mltgnt.memory._iterative — iterative retrieval using judge_sufficiency.

Design: Issue #913
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

_log = logging.getLogger(__name__)

SkillSearchFn = Callable[[str, int], list]  # (query, max_entries) -> list[ScoredEntry]

__all__ = [
    "SearchResult",
    "IterativeRetriever",
    "SkillSearchFn",
]


@dataclass(frozen=True)
class SearchResult:
    """One search result."""

    source: Literal["memory", "skill"]
    entries: list  # list[ScoredEntry]


class IterativeRetriever:
    """Retriever that gathers info iteratively via judge_sufficiency.

    Each time the LLM judges INSUFFICIENT, re-search memory or skill by source
    until SUFFICIENT or max_iterations.
    """

    def __init__(
        self,
        config: "MemoryConfig",
        persona_stem: str,
        llm_call: Callable[[str], str],
        *,
        search_skills: SkillSearchFn | None = None,
        max_iterations: int = 3,
    ) -> None:
        self._config = config
        self._persona_stem = persona_stem
        self._search_skills = search_skills
        self._llm_call = llm_call
        self._max_iterations = max_iterations

    def retrieve(self, query: str, *, max_bytes: int, max_entries: int) -> str:
        """Run the iterative search loop; return collected info as text.

        Returns:
            preferences + collected entries joined (within max_bytes)
        """
        from mltgnt.memory._sufficiency import judge_sufficiency

        # Step 0: initial search from memory
        initial_entries = self._search_memory(query, max_entries)

        # Collected entries (dedupe by text key)
        collected: dict[str, object] = {e.text: e for e in initial_entries}

        # Loop (up to max_iterations)
        for _ in range(self._max_iterations):
            collected_text = self._format_collected(collected)

            try:
                result = judge_sufficiency(query, collected_text, self._llm_call)
            except Exception as e:
                _log.warning(
                    "IterativeRetriever: LLM call failed, returning collected results: %s", e
                )
                break

            if result.sufficient:
                break

            action = result.action
            if action is None:
                break

            # Action: search by source
            if action.source == "memory":
                new_entries = self._search_memory(action.query, max_entries)
            else:  # "skill"
                if self._search_skills is None:
                    new_entries = []
                else:
                    new_entries = self._search_skills(action.query, max_entries)

            # Observe: merge new entries into collected (dedupe)
            for entry in new_entries:
                if entry.text not in collected:
                    collected[entry.text] = entry

        return self._build_output(collected, max_bytes, max_entries)

    def retrieve_skills(self, query: str, max_entries: int) -> list:
        """Search the skill source only; return ScoredEntry list.

        Return [] if search_skills callback is not injected.
        """
        if self._search_skills is None:
            return []
        return self._search_skills(query, max_entries)

    def _search_memory(self, query: str, max_entries: int) -> list:
        """Score memory-file entries with TF-IDF (JSONL-aware).

        Returns:
            list[ScoredEntry]
        """
        from mltgnt.memory._scoring import score_entries, ScoredEntry
        from mltgnt.memory.api import _ensure_jsonl
        from mltgnt.memory._format import parse_jsonl, assemble_entries_text

        jsonl_path = _ensure_jsonl(self._config, self._persona_stem)
        if not jsonl_path.exists():
            return []

        entries = parse_jsonl(jsonl_path)
        non_prefs = [e for e in entries if e.source_tag != "preferences"]

        if not non_prefs:
            return []

        entry_texts = [
            assemble_entries_text(
                [e],
                preferences_heading=self._config.preferences_section_name,
            ).strip()
            for e in non_prefs
        ]

        if not query:
            return [ScoredEntry(text=t, score=0.0) for t in entry_texts[-max_entries:]]

        scored = score_entries(query, entry_texts)
        return scored[:max_entries]

    def _format_collected(self, collected: dict) -> str:
        """Join collected entries as text."""
        if not collected:
            return ""
        return "\n\n---\n\n".join(e.text for e in collected.values())

    def _build_output(self, collected: dict, max_bytes: int, max_entries: int) -> str:
        """Join preferences + collected entries within max_bytes."""
        from mltgnt.memory.api import read_memory_preferences

        prefs = read_memory_preferences(self._config, self._persona_stem)

        # Take top max_entries by score descending
        sorted_entries = sorted(
            collected.values(), key=lambda e: e.score, reverse=True
        )
        top_entries = [e.text for e in sorted_entries[:max_entries]]

        parts: list[str] = []
        if prefs:
            parts.append(prefs)
        parts.extend(top_entries)

        if not parts:
            return prefs  # may be empty string

        text = "\n\n---\n\n".join(parts)

        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
