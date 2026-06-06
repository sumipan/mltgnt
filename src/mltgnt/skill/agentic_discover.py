"""
mltgnt.skill.agentic_discover — AgenticSkillDiscoverer によるスキル発見。

設計: Issue #1895 Subtask 1 / Issue #1922
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from mltgnt.memory._sufficiency import judge_for_discover
from mltgnt.skill.models import SkillMeta

__all__ = [
    "AgenticSkillDiscoverer",
    "DiscoverResult",
    "DiscoverRound",
]


@dataclass(frozen=True)
class DiscoverRound:
    """1 ラウンド分のスキル発見履歴。"""

    query: str
    candidates: list[tuple[str, float]]  # (skill_name, score)
    verdict: str  # "SELECTED" | "NEED_MORE" | "UNRESOLVED"


@dataclass(frozen=True)
class DiscoverResult:
    """スキル発見の最終結果。"""

    kind: Literal["selected", "ambiguous", "unresolved"]
    skill: SkillMeta | None = None
    candidates: list[tuple[SkillMeta, float]] = field(default_factory=list)
    trace: list[DiscoverRound] = field(default_factory=list)


class AgenticSkillDiscoverer:
    """TF-IDF スコアリングと judge_for_discover を反復ループで組み合わせ、
    スキルカタログから候補を絞り込む。
    """

    def __init__(
        self,
        llm_call: Callable[[str], str],
        *,
        max_iterations: int = 3,
    ) -> None:
        self._llm_call = llm_call
        self._max_iterations = max_iterations

    def discover(
        self,
        user_input: str,
        skill_catalog: dict[str, SkillMeta],
        persona_skills: list[str] | None = None,
    ) -> DiscoverResult:
        catalog = self._filter_catalog(skill_catalog, persona_skills)
        if not catalog:
            return DiscoverResult(kind="unresolved")

        trace: list[DiscoverRound] = []
        query = user_input
        last_candidates: list[tuple[str, float]] = []

        for _ in range(self._max_iterations):
            scored = self._score_catalog(query, catalog, top_n=5)
            last_candidates = scored
            if not scored:
                break

            collected_text = self._format_candidates(scored, catalog)
            skill_names = [name for name, _ in scored]
            verdict = judge_for_discover(
                query, collected_text, skill_names, self._llm_call
            )

            if verdict.kind == "selected":
                round_verdict = "SELECTED"
            elif verdict.kind == "need_more":
                round_verdict = "NEED_MORE"
            else:
                round_verdict = "UNRESOLVED"

            trace.append(
                DiscoverRound(query=query, candidates=scored, verdict=round_verdict)
            )

            if verdict.kind == "selected":
                skill = (
                    catalog.get(verdict.skill_name)
                    if verdict.skill_name is not None
                    else None
                )
                return DiscoverResult(
                    kind="selected",
                    skill=skill,
                    trace=trace,
                )

            if verdict.kind == "unresolved":
                return DiscoverResult(
                    kind="unresolved",
                    candidates=self._to_meta_candidates(scored, catalog),
                    trace=trace,
                )

            query = verdict.next_query or user_input

        if last_candidates:
            return DiscoverResult(
                kind="ambiguous",
                candidates=self._to_meta_candidates(last_candidates, catalog),
                trace=trace,
            )

        return DiscoverResult(kind="unresolved", trace=trace)

    @staticmethod
    def _filter_catalog(
        skill_catalog: dict[str, SkillMeta],
        persona_skills: list[str] | None,
    ) -> dict[str, SkillMeta]:
        if persona_skills is None:
            return skill_catalog
        return {k: v for k, v in skill_catalog.items() if k in persona_skills}

    @staticmethod
    def _format_candidates(
        scored: list[tuple[str, float]],
        catalog: dict[str, SkillMeta],
    ) -> str:
        lines: list[str] = []
        for name, score in scored:
            meta = catalog[name]
            lines.append(f"{name}: {meta.description} (score: {score:.2f})")
        return "\n".join(lines)

    @staticmethod
    def _to_meta_candidates(
        scored: list[tuple[str, float]],
        catalog: dict[str, SkillMeta],
    ) -> list[tuple[SkillMeta, float]]:
        return [
            (catalog[name], score) for name, score in scored if name in catalog
        ]

    @staticmethod
    def _score_catalog(
        query: str,
        catalog: dict[str, SkillMeta],
        *,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        from mltgnt.memory._scoring import score_entries

        if not catalog:
            return []

        names = list(catalog.keys())
        texts = [
            catalog[name].description + " " + " ".join(catalog[name].triggers)
            for name in names
        ]
        text_to_name = dict(zip(texts, names))
        scored_entries = score_entries(query, texts)
        result: list[tuple[str, float]] = []
        for entry in scored_entries[:top_n]:
            name = text_to_name.get(entry.text)
            if name is not None:
                result.append((name, entry.score))
        return result
