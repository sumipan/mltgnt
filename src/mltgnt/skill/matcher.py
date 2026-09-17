"""
mltgnt.skill.matcher — skill matching (slash / literal / triggers / LLM).

Design: Issue #124 §6.3, Issue #208, Issue #1384 U5
"""
from __future__ import annotations

import logging
import re

from mltgnt.bridges.llm_adapter import call_llm as llm_call
from mltgnt.routing.agentic_discover import AgenticSkillDiscoverer, DiscoverResult

from mltgnt.skill.models import SkillMatchResult, SkillMeta

_log = logging.getLogger(__name__)

_SLASH_PATTERN = re.compile(r"^/(\S+)(.*)", re.DOTALL)

_DEFAULT_MATCHER_MODEL = "claude-haiku-4-5-20251001"

_LLM_SYSTEM_PROMPT = """\
You are a skill matcher.
Decide whether the user input corresponds to one of the skills below.
If a skill matches, return only that skill name.
If none match, return only "none".
No extra explanation.
"""


def _filter_by_persona(
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> dict[str, SkillMeta]:
    if persona_skills is None:
        return skills
    return {k: v for k, v in skills.items() if k in persona_skills}


def _match_by_literal(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> SkillMatchResult | None:
    """Find a skill by literal name match.

    Return SkillMatchResult only on a single hit. None on multi-hit or miss.
    """
    filtered = _filter_by_persona(skills, persona_skills)
    hits = [meta for meta in filtered.values() if meta.name in user_input]
    if len(hits) == 1:
        meta = hits[0]
        return SkillMatchResult(
            decisive=meta,
            candidates=[],
            rationale=f"literal:{meta.name}",
            arguments=user_input,
        )
    return None


def _match_by_triggers(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> tuple[SkillMeta, str] | None:
    """Find a skill by partial trigger-keyword match.

    Args:
        user_input: User input string
        skills: Skill dict from discover()
        persona_skills: Allowed skill names for the persona (None = unrestricted)

    Returns:
        (SkillMeta, user_input) or None
        On trigger match, pass the full user_input as arguments
    """
    filtered = _filter_by_persona(skills, persona_skills)
    for meta in filtered.values():
        for trigger in meta.triggers:
            if trigger in user_input:
                return (meta, user_input)
    return None


def match_triggers_only(user_input: str, skills: dict[str, SkillMeta]) -> str | None:
    """Find a skill by trigger keywords only (no persona filter).

    For lightweight matching (e.g. knowledge recording) where LLM fallback is unnecessary.
    When a persona filter is needed, the caller should pre-filter skills.

    Returns:
        Matched skill name, or None
    """
    for meta in skills.values():
        for trigger in meta.triggers:
            if trigger in user_input:
                return meta.name
    return None


def _trigger_rationale(
    user_input: str,
    meta: SkillMeta,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
) -> str:
    filtered = _filter_by_persona(skills, persona_skills)
    for candidate in filtered.values():
        if candidate.name != meta.name:
            continue
        for trigger in candidate.triggers:
            if trigger in user_input:
                return f"trigger:{trigger}"
    return "trigger:unknown"


async def _match_by_llm(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None,
    model: str | None = None,
) -> tuple[SkillMeta, str] | None:
    """Pass the skill list and input to an LLM for intent classification.

    Args:
        user_input: User input string
        skills: Skill dict from discover()
        persona_skills: Allowed skill names for the persona (None = unrestricted)

    Returns:
        (SkillMeta, user_input) or None
        None when the LLM returns "none" or a name not in registered skills
    """
    filtered = _filter_by_persona(skills, persona_skills)
    if not filtered:
        return None

    skill_list = "\n".join(
        f"- {meta.name}: {meta.description}" for meta in filtered.values()
    )
    prompt = f"{_LLM_SYSTEM_PROMPT}\n\nSkill list:\n{skill_list}\n\nUser input: {user_input}"

    try:
        result = llm_call(prompt, engine="claude", model=model or _DEFAULT_MATCHER_MODEL, timeout=30)
        if not result.success:
            _log.warning("LLM intent classification error: %s", result.stderr)
            return None
        response = result.body.strip().lower()
    except Exception as e:
        _log.warning("LLM intent classification error: %s", e)
        return None

    if response == "none" or response not in filtered:
        return None

    return (filtered[response], user_input)


def split_pipe_segments(user_input: str) -> list[str]:
    """Split user input on ' | ' (spaces required).

    Returns:
        List of segments. Single-element list when there is no pipe.
    """
    return user_input.split(" | ")


async def match_pipeline(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None = None,
    model: str | None = None,
) -> list[SkillMatchResult]:
    """Split pipe input and delegate each segment to match().

    Returns:
        List of match() results in input order. Single element when not piped.
    """
    segments = split_pipe_segments(user_input)
    results: list[SkillMatchResult] = []
    for segment in segments:
        result = await match(segment, skills, persona_skills=persona_skills, model=model)
        results.append(result)
    return results


async def match(
    user_input: str,
    skills: dict[str, SkillMeta],
    persona_skills: list[str] | None = None,
    model: str | None = None,
) -> SkillMatchResult:
    """
    Identify a skill from user input (5-stage fallback).

    Priority: slash command -> literal match -> trigger substring
    -> AgenticSkillDiscoverer -> LLM intent classification

    Returns: SkillMatchResult. decisive=None, rationale="none" when unmatched.
    """
    # Step 1: slash command
    m = _SLASH_PATTERN.match(user_input)
    if m:
        name = m.group(1)
        rest = m.group(2)
        arguments = rest.lstrip(" ") if rest else ""
        if name not in skills:
            return SkillMatchResult(
                decisive=None,
                candidates=[],
                rationale="none",
                arguments=arguments,
            )
        meta = skills[name]
        if persona_skills is not None and name not in persona_skills:
            return SkillMatchResult(
                decisive=None,
                candidates=[],
                rationale="none",
                arguments=arguments,
            )
        return SkillMatchResult(
            decisive=meta,
            candidates=[],
            rationale=f"slash:{name}",
            arguments=arguments,
        )

    # Step 2: literal match
    literal_result = _match_by_literal(user_input, skills, persona_skills)
    if literal_result is not None:
        return literal_result

    # Step 3: trigger substring
    trigger_result = _match_by_triggers(user_input, skills, persona_skills)
    if trigger_result is not None:
        meta, arguments = trigger_result
        return SkillMatchResult(
            decisive=meta,
            candidates=[],
            rationale=_trigger_rationale(user_input, meta, skills, persona_skills),
            arguments=arguments,
        )

    # Step 4: AgenticSkillDiscoverer (iterative narrowing)
    def _llm_for_discover(prompt: str) -> str:
        result = llm_call(
            prompt, engine="claude", model=model or _DEFAULT_MATCHER_MODEL, timeout=30
        )
        if not result.success:
            raise RuntimeError(result.stderr)
        return str(result.body or "").strip()

    try:
        discoverer = AgenticSkillDiscoverer(llm_call=_llm_for_discover, max_iterations=3)
        discover_result = discoverer.discover(user_input, skills, persona_skills=persona_skills)
    except Exception:
        _log.warning("AgenticSkillDiscoverer failed, falling back to LLM", exc_info=True)
        discover_result = DiscoverResult(kind="unresolved")

    if discover_result.kind == "selected" and discover_result.skill is not None:
        return SkillMatchResult(
            decisive=discover_result.skill,
            candidates=[],
            rationale=f"agentic:{discover_result.skill.name}",
            arguments=user_input,
        )
    if discover_result.kind == "ambiguous" and discover_result.candidates:
        top_skill, _score = discover_result.candidates[0]
        return SkillMatchResult(
            decisive=top_skill,
            candidates=[],
            rationale=f"agentic-ambiguous:{top_skill.name}",
            arguments=user_input,
        )

    # Step 5: existing LLM intent-classification fallback (when unresolved)
    llm_result = await _match_by_llm(user_input, skills, persona_skills, model=model)
    if llm_result is not None:
        meta, arguments = llm_result
        return SkillMatchResult(
            decisive=meta,
            candidates=[],
            rationale=f"llm:{meta.name}",
            arguments=arguments,
        )

    return SkillMatchResult(
        decisive=None,
        candidates=[],
        rationale="none",
        arguments=user_input,
    )
