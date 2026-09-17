"""Preflight skeleton before dispatch (#3318).

Memory, skill, profile, and note implementations are host-injected.
Prompt bodies and policy are not hardcoded in this module.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone as _timezone
from typing import Any, Callable

_log = logging.getLogger(__name__)

_SKILL_ESTIMATE_THREAD_TAIL_CHARS = 1200


@dataclass(frozen=True)
class MemoryWorkerResult:
    excerpt: str | None
    start_ts: str
    duration_ms: int
    status: str


@dataclass(frozen=True)
class SkillWorkerResult:
    persona_skills: list[str] | None
    selected_skill: str | None
    start_ts: str
    duration_ms: int
    status: str
    rationale: str | None = None


@dataclass(frozen=True)
class PreflightContext:
    triage_profile_content: str | None
    memory_excerpt: str | None
    memory_worker: MemoryWorkerResult
    skill_worker: SkillWorkerResult
    secretary_memo: str | None
    user_preferences: str | None
    thread_context: str | None
    preflight_note: str | None

    @property
    def persona_skills(self) -> list[str] | None:
        return self.skill_worker.persona_skills

    @property
    def selected_skill(self) -> str | None:
        return self.skill_worker.selected_skill

    @property
    def selected_skill_is_deterministic(self) -> bool:
        rationale = self.skill_worker.rationale
        if not rationale:
            return False
        return rationale.startswith("literal:") or rationale.startswith("trigger:")


def _utc_now() -> str:
    return datetime.now(_timezone.utc).isoformat()


def _build_thread_tail(
    thread_messages: list[dict] | None,
    *,
    own_bot_id: str | None,
    persona_name: str,
    format_thread_for_prompt_fn: Callable[..., str | None],
) -> str | None:
    if not thread_messages:
        return None
    tail_msgs = thread_messages[-3:]
    formatted = format_thread_for_prompt_fn(
        tail_msgs,
        own_bot_id=own_bot_id,
        persona_name=persona_name,
    )
    if not formatted:
        return None
    if len(formatted) > _SKILL_ESTIMATE_THREAD_TAIL_CHARS:
        return formatted[-_SKILL_ESTIMATE_THREAD_TAIL_CHARS:]
    return formatted


def run_preflight(
    *,
    effective_persona: str,
    instruction: str,
    thread_key: str,
    profile_content: str | None,
    routing_result: Any,
    thread_messages: list[dict] | None,
    load_persona_fn: Callable[..., tuple[str | None, Any, Any]],
    memory_worker_fn: Callable[[str, str, str], MemoryWorkerResult],
    skill_worker_fn: Callable[[str, str, list[str], str | None], SkillWorkerResult],
    read_secretary_memo_fn: Callable[[], str | None],
    read_memory_preferences_fn: Callable[[str], str | None],
    format_thread_for_prompt_fn: Callable[..., str | None],
    build_preflight_note_fn: Callable[[str], str | None],
    own_bot_id: str | None = None,
    logger: logging.Logger | None = None,
) -> PreflightContext:
    """Run parallel preflight before dispatch. All workers must be injected."""
    triage_profile_content = profile_content
    _logger = logger or _log
    try:
        light_profile, _, _ = load_persona_fn(effective_persona, weight="light")
        if light_profile:
            triage_profile_content = light_profile
    except Exception:
        _logger.warning(
            "[dispatch] failed to load light profile for triage persona=%s",
            effective_persona,
            exc_info=True,
        )

    thread_context: str | None = None
    if thread_messages:
        thread_context = format_thread_for_prompt_fn(
            thread_messages,
            own_bot_id=own_bot_id,
            persona_name=effective_persona,
        )
    thread_tail = _build_thread_tail(
        thread_messages,
        own_bot_id=own_bot_id,
        persona_name=effective_persona,
        format_thread_for_prompt_fn=format_thread_for_prompt_fn,
    )

    prebound_copy = list((routing_result.prebound_skills if routing_result else None) or [])
    with ThreadPoolExecutor(max_workers=2) as preflight_pool:
        mem_fut = preflight_pool.submit(
            memory_worker_fn, effective_persona, thread_key, instruction
        )
        skill_fut = preflight_pool.submit(
            skill_worker_fn,
            effective_persona,
            instruction,
            prebound_copy,
            thread_tail,
        )
        mem_result = mem_fut.result()
        skill_result = skill_fut.result()

    user_profile = read_memory_preferences_fn(effective_persona) or None

    return PreflightContext(
        triage_profile_content=triage_profile_content,
        memory_excerpt=mem_result.excerpt,
        memory_worker=mem_result,
        skill_worker=skill_result,
        secretary_memo=read_secretary_memo_fn(),
        user_preferences=user_profile,
        thread_context=thread_context,
        preflight_note=build_preflight_note_fn(instruction),
    )


# Re-export for tests and host compatibility
utc_now = _utc_now
build_thread_tail = _build_thread_tail
SKILL_ESTIMATE_THREAD_TAIL_CHARS = _SKILL_ESTIMATE_THREAD_TAIL_CHARS
