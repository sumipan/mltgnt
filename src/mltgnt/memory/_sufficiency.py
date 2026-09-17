"""
mltgnt.memory._sufficiency — LLM-based sufficiency judgment.

Design: Issue #197 Phase 2/3
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

_log = logging.getLogger(__name__)

SUFFICIENT_TOKEN = "SUFFICIENT"
INSUFFICIENT_TOKEN = "INSUFFICIENT"
MEMORY_TOKEN = "MEMORY"
SKILL_TOKEN = "SKILL"

SELECTED_TOKEN = "SELECTED"
NEED_MORE_TOKEN = "NEED_MORE"
UNRESOLVED_TOKEN = "UNRESOLVED"

__all__ = [
    "DiscoverVerdict",
    "SearchAction",
    "SufficiencyResult",
    "judge_for_discover",
    "judge_sufficiency",
]


@dataclass(frozen=True)
class SearchAction:
    """Next search action decided by the LLM."""

    source: Literal["memory", "skill"]
    query: str


@dataclass(frozen=True)
class DiscoverVerdict:
    """Three-way verdict for skill discovery."""

    kind: Literal["selected", "need_more", "unresolved"]
    skill_name: str | None = None  # non-None only when selected
    next_query: str | None = None  # non-None only when need_more
    reason: str | None = None  # non-None only when unresolved
    top_candidates: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class SufficiencyResult:
    """Sufficiency judgment result."""

    sufficient: bool
    action: SearchAction | None  # None when sufficient=True

    @property
    def rewritten_query(self) -> str | None:
        """Phase 2 compat property. Delegates to action.query."""
        if self.action is None:
            return None
        return self.action.query


def _build_prompt(query: str, collected_text: str) -> str:
    return f"""You are an assistant that judges whether collected information is sufficient.

User question: {query}

Collected information:
{collected_text}

Judge whether the information above is enough to answer the user's question.

If sufficient, output only "SUFFICIENT" on line 1.
If insufficient, output in this format:
Line 1: "INSUFFICIENT"
Line 2: search source ("MEMORY" or "SKILL")
Line 3: a search query to fill the missing information"""


def judge_sufficiency(
    query: str,
    collected_text: str,
    llm_call: Callable[[str], str],
) -> SufficiencyResult:
    """Judge sufficiency of collected information using an LLM.

    On parse failure, treat as sufficient=True (fail-safe).
    If the LLM call itself raises, propagate to the caller.

    Args:
        query: User question
        collected_text: Collected information text
        llm_call: Function that calls the LLM

    Returns:
        SufficiencyResult
    """
    prompt = _build_prompt(query, collected_text)
    response = llm_call(prompt)

    lines = [line.strip() for line in response.strip().splitlines()]
    if not lines:
        _log.warning("judge_sufficiency: empty response, treating as SUFFICIENT")
        return SufficiencyResult(sufficient=True, action=None)

    first = lines[0]

    if first == SUFFICIENT_TOKEN:
        return SufficiencyResult(sufficient=True, action=None)

    if first == INSUFFICIENT_TOKEN:
        if len(lines) < 3:
            _log.warning(
                "judge_sufficiency: INSUFFICIENT response missing source/query lines, "
                "treating as SUFFICIENT"
            )
            return SufficiencyResult(sufficient=True, action=None)
        source_raw = lines[1].upper()
        requery = lines[2]
        if source_raw == MEMORY_TOKEN:
            source: Literal["memory", "skill"] = "memory"
        elif source_raw == SKILL_TOKEN:
            source = "skill"
        else:
            _log.warning(
                "judge_sufficiency: unknown source '%s', treating as SUFFICIENT",
                source_raw,
            )
            return SufficiencyResult(sufficient=True, action=None)
        return SufficiencyResult(
            sufficient=False,
            action=SearchAction(source=source, query=requery),
        )

    # Unknown format → fail-safe
    _log.warning(
        "judge_sufficiency: unexpected response format '%s', treating as SUFFICIENT",
        first,
    )
    return SufficiencyResult(sufficient=True, action=None)


def _build_discover_prompt(
    query: str,
    collected_text: str,
    skill_names: list[str],
) -> str:
    skills_list = ", ".join(skill_names)
    return f"""You are an assistant that picks the single skill that best matches the user intent.

User input: {query}

Candidate skills:
{collected_text}

Candidate skill names: {skills_list}

Choose the best skill, ask for more information, or decide none apply.

When narrowed to one:
Line 1: SELECTED
Line 2: skill name

When more search is needed:
Line 1: NEED_MORE
Line 2: a refining search query

When none apply or undecidable:
Line 1: UNRESOLVED"""


def judge_for_discover(
    query: str,
    collected_text: str,
    skill_names: list[str],
    llm_call: Callable[[str], str],
) -> DiscoverVerdict:
    """Three-way verdict over skill candidates using an LLM.

    On parse failure, return unresolved (safe side).
    If the LLM call itself raises, propagate to the caller.
    """
    prompt = _build_discover_prompt(query, collected_text, skill_names)
    response = llm_call(prompt)

    lines = [line.strip() for line in response.strip().splitlines()]
    if not lines:
        _log.warning("judge_for_discover: empty response, treating as UNRESOLVED")
        return DiscoverVerdict(kind="unresolved", reason="parse_error")

    first = lines[0]

    if first == SELECTED_TOKEN:
        if len(lines) < 2 or not lines[1]:
            _log.warning(
                "judge_for_discover: SELECTED response missing skill name, "
                "treating as UNRESOLVED"
            )
            return DiscoverVerdict(kind="unresolved", reason="parse_error")
        return DiscoverVerdict(kind="selected", skill_name=lines[1])

    if first == NEED_MORE_TOKEN:
        if len(lines) < 2 or not lines[1]:
            _log.warning(
                "judge_for_discover: NEED_MORE response missing query, "
                "treating as UNRESOLVED"
            )
            return DiscoverVerdict(kind="unresolved", reason="parse_error")
        return DiscoverVerdict(kind="need_more", next_query=lines[1])

    if first == UNRESOLVED_TOKEN:
        return DiscoverVerdict(kind="unresolved", reason="no_match")

    _log.warning(
        "judge_for_discover: unexpected response format '%s', treating as UNRESOLVED",
        first,
    )
    return DiscoverVerdict(kind="unresolved", reason="parse_error")
