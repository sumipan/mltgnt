"""
mltgnt.memory._agentic — Plan→Action→Observe ループによる Agentic RAG。

設計: Issue #200 Phase 3
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mltgnt.config import MemoryConfig

_log = logging.getLogger(__name__)

__all__ = [
    "SearchResult",
    "AgenticRetriever",
]


@dataclass(frozen=True)
class SearchResult:
    """1回の検索結果"""

    source: Literal["memory", "skill"]
    entries: list  # list[ScoredEntry]


def _search_skills(
    query: str,
    skill_paths: list[Path],
    max_entries: int,
) -> list:
    """skill ディレクトリの SKILL.md 本文を TF-IDF 検索する。

    Returns:
        list[ScoredEntry]
    """
    if not skill_paths:
        return []
    from mltgnt.skill.loader import discover
    from mltgnt.memory._scoring import score_entries

    skills = discover(skill_paths)
    if not skills:
        return []
    bodies = [Path(meta.path).read_text(encoding="utf-8") for meta in skills.values()]
    if not bodies:
        return []
    scored = score_entries(query, bodies)
    return scored[:max_entries]


class AgenticRetriever:
    """Plan→Action→Observe ループを実行するエージェント型リトリーバー。"""

    def __init__(
        self,
        config: "MemoryConfig",
        persona_stem: str,
        skill_paths: list[Path],
        llm_call: Callable[[str], str],
        *,
        max_iterations: int = 3,
    ) -> None:
        self._config = config
        self._persona_stem = persona_stem
        self._skill_paths = skill_paths
        self._llm_call = llm_call
        self._max_iterations = max_iterations

    def retrieve(self, query: str, *, max_bytes: int, max_entries: int) -> str:
        """Plan→Action→Observe ループを実行し、収集した情報をテキストとして返す。

        Returns:
            preferences + 収集エントリを結合したテキスト（max_bytes 以内）
        """
        from mltgnt.memory._sufficiency import judge_sufficiency

        # Step 0: memory から初回検索
        initial_entries = self._search_memory(query, max_entries)

        # 収集済みエントリ（テキストをキーとして重複排除）
        collected: dict[str, object] = {e.text: e for e in initial_entries}

        # LLM 呼び出し失敗フラグ
        llm_failed = False

        # Loop (最大 max_iterations 回)
        for _ in range(self._max_iterations):
            collected_text = self._format_collected(collected)

            try:
                result = judge_sufficiency(query, collected_text, self._llm_call)
            except Exception as e:
                _log.warning(
                    "AgenticRetriever: LLM call failed, returning collected results: %s", e
                )
                llm_failed = True
                break

            if result.sufficient:
                break

            action = result.action
            assert action is not None

            # Action: source に応じて検索実行
            if action.source == "memory":
                new_entries = self._search_memory(action.query, max_entries)
            else:  # "skill"
                new_entries = _search_skills(action.query, self._skill_paths, max_entries)

            # Observe: 新規エントリを collected にマージ（重複排除）
            for entry in new_entries:
                if entry.text not in collected:
                    collected[entry.text] = entry

        return self._build_output(collected, max_bytes, max_entries)

    def _search_memory(self, query: str, max_entries: int) -> list:
        """memory ファイルからエントリを TF-IDF でスコアリングして返す。

        Returns:
            list[ScoredEntry]
        """
        from mltgnt.memory._scoring import score_entries, ScoredEntry
        from mltgnt.memory import memory_file_path

        path = memory_file_path(self._config, self._persona_stem)
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        if not raw.strip():
            return []

        prefs_header = "## ユーザーの好み・傾向"
        parts = re.split(r"\n---\s*\n", raw)
        entry_blocks = [
            p.strip() for p in parts if p.strip() and prefs_header not in p
        ]

        if not entry_blocks:
            return []

        if not query:
            return [ScoredEntry(text=b, score=0.0) for b in entry_blocks[-max_entries:]]

        scored = score_entries(query, entry_blocks)
        return scored[:max_entries]

    def _format_collected(self, collected: dict) -> str:
        """収集済みエントリをテキストとして結合する。"""
        if not collected:
            return ""
        return "\n\n---\n\n".join(e.text for e in collected.values())

    def _build_output(self, collected: dict, max_bytes: int, max_entries: int) -> str:
        """preferences + 収集エントリを結合して max_bytes 以内で返す。"""
        from mltgnt.memory import read_memory_preferences

        prefs = read_memory_preferences(self._config, self._persona_stem)

        # スコア降順で上位 max_entries を選択
        sorted_entries = sorted(
            collected.values(), key=lambda e: e.score, reverse=True
        )
        top_entries = [e.text for e in sorted_entries[:max_entries]]

        parts: list[str] = []
        if prefs:
            parts.append(prefs)
        parts.extend(top_entries)

        if not parts:
            return prefs  # 空文字列かもしれない

        text = "\n\n---\n\n".join(parts)

        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
