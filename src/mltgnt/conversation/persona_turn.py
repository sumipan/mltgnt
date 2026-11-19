"""Standard one-turn persona conversation flow (#7434).

``PersonaTurnHandler`` loads a persona, assembles memory and history into a
prompt, runs the LLM via DAG ``enqueue_and_wait``, optionally records episodes,
and returns a ``TurnResult``. Hosts inject routing (``resolve_persona``),
static instructions, per-turn extensions, and reply post-processing.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from mltgnt.bridges.ghdag_bridge import Permission, enqueue_and_wait
from mltgnt.config import MemoryConfig
from mltgnt.interfaces.turn import TurnInput, TurnResult
from mltgnt.media._core import id_map
from mltgnt.memory import append_turn_episodes, build_memory_context
from mltgnt.memory.api import memory_file_path
from mltgnt.persona import load_persona

_log = logging.getLogger(__name__)

__all__ = ["PersonaTurnConfig", "PersonaTurnHandler"]

EnqueueFn = Callable[..., tuple[bool, str]]
ResolvePersonaFn = Callable[[TurnInput], str]
ExtendPromptFn = Callable[[TurnInput], str]
PostprocessFn = Callable[[TurnInput, str], TurnResult | str]
NowFn = Callable[[], datetime]


@dataclass(frozen=True)
class PersonaTurnConfig:
    """Paths and memory/LLM options for :class:`PersonaTurnHandler`."""

    persona_dir: Path
    jobs_dir: Path
    exec_done_dir: Path
    memory: MemoryConfig | None = None
    timeout_sec: float = 300
    history_turns: int = 20
    core_max_bytes: int = 4096
    thread_tail_entries: int = 0
    recent_scope: Literal["thread", "peer", "persona"] = "thread"
    recent_hours: int = 48
    recent_entries: int = 12
    recent_per_thread_entries: int = 4
    permission: str | Permission | None = None


class PersonaTurnHandler:
    """TurnHandler that runs load → memory → LLM (DAG) → record for one turn.

    Minimal wiring::

        handler = PersonaTurnHandler(
            PersonaTurnConfig(persona_dir=..., jobs_dir=..., exec_done_dir=...),
            resolve_persona=lambda turn: "persona-a",
            instruction="Reply briefly.",
        )
        result = handler.handle(turn)

    ``recent_scope`` controls cross-thread episodic context: ``thread`` (default)
    omits other threads; ``peer`` includes the same ``author_id`` only; ``persona``
    includes all recent threads. For multi-user channels use ``peer``; for a
    single user, ``persona`` is often enough.
    """

    def __init__(
        self,
        config: PersonaTurnConfig,
        *,
        resolve_persona: ResolvePersonaFn,
        instruction: str | None = None,
        extend_prompt: ExtendPromptFn | None = None,
        postprocess: PostprocessFn | None = None,
        enqueue: EnqueueFn = enqueue_and_wait,
        now: NowFn | None = None,
    ) -> None:
        self._config = config
        self._resolve_persona = resolve_persona
        self._instruction = instruction
        self._extend_prompt = extend_prompt
        self._postprocess = postprocess
        self._enqueue = enqueue
        self._now = now or (lambda: datetime.now(timezone.utc))

    def handle(self, turn: TurnInput) -> TurnResult:
        space_id, thread_key = id_map.resolve(turn.conversation_id)
        name = self._resolve_persona(turn)
        persona = load_persona(name, persona_dir=self._config.persona_dir)

        memory_context = self._build_memory_context(
            persona.name,
            space_id,
            thread_key,
            turn,
        )
        prompt = self._build_prompt(persona, turn, memory_context)
        turn_id = uuid4().hex
        engine = persona.fm.engine or "claude"
        model = persona.fm.model or None

        ok, content = self._enqueue(
            prompt,
            engine,
            model,
            self._config.timeout_sec,
            f"turn:{turn.conversation_id}:{turn_id}",
            self._config.jobs_dir,
            self._config.exec_done_dir,
            persona_name=persona.name,
            persona_dir=self._config.persona_dir,
            permission=self._config.permission,
        )
        if not ok:
            return TurnResult(kind="reply", text=f"(error: {content})")

        result = self._apply_postprocess(turn, content)
        self._record_episode(
            persona.name,
            turn,
            result,
            space_id,
            thread_key,
            turn_id,
        )
        return result

    def _memory_layout_dir(self, persona_stem: str) -> Path | None:
        memory_cfg = self._config.memory
        if memory_cfg is None:
            return None
        path = memory_file_path(memory_cfg, persona_stem)
        if path.name == "episodes.jsonl" and path.parent.is_dir():
            return path.parent
        return None

    def _build_memory_context(
        self,
        persona_stem: str,
        space_id: str,
        thread_key: str,
        turn: TurnInput,
    ) -> str:
        mem_dir = self._memory_layout_dir(persona_stem)
        if mem_dir is None:
            return ""

        recent_hours = 0
        recent_entries = 0
        recent_peer: str | None = None
        recent_per_thread = self._config.recent_per_thread_entries

        scope = self._config.recent_scope
        if scope == "persona":
            recent_hours = self._config.recent_hours
            recent_entries = self._config.recent_entries
        elif scope == "peer" and turn.author_id is not None:
            recent_hours = self._config.recent_hours
            recent_entries = self._config.recent_entries
            recent_peer = turn.author_id

        return build_memory_context(
            mem_dir,
            persona_stem=persona_stem,
            space_id=space_id,
            thread_key=thread_key,
            core_max_bytes=self._config.core_max_bytes,
            thread_tail_entries=self._config.thread_tail_entries,
            recent_hours=recent_hours,
            recent_entries=recent_entries,
            recent_peer=recent_peer,
            recent_per_thread_entries=recent_per_thread,
            now=self._now(),
        )

    def _build_prompt(self, persona, turn: TurnInput, memory_context: str) -> str:
        parts: list[str] = []
        if memory_context:
            parts.append(memory_context)
        if self._config.history_turns > 0 and turn.history:
            lines = [
                f"{msg.role}: {msg.text}"
                for msg in turn.history[-self._config.history_turns :]
            ]
            if lines:
                parts.append("\n".join(lines))
        if turn.text:
            parts.append(turn.text)
        if self._instruction:
            parts.append(self._instruction)
        if self._extend_prompt is not None:
            extra = self._extend_prompt(turn)
            if extra:
                parts.append(extra)
        combined = "\n\n".join(parts)
        return persona.format_prompt(combined)

    def _apply_postprocess(self, turn: TurnInput, reply_text: str) -> TurnResult:
        if self._postprocess is None:
            return TurnResult(kind="reply", text=reply_text)
        out = self._postprocess(turn, reply_text)
        if isinstance(out, str):
            return TurnResult(kind="reply", text=out)
        return out

    def _record_episode(
        self,
        persona_stem: str,
        turn: TurnInput,
        result: TurnResult,
        space_id: str,
        thread_key: str,
        turn_id: str,
    ) -> None:
        memory_cfg = self._config.memory
        if memory_cfg is None or not result.text:
            return
        ts = self._now().isoformat(timespec="seconds")
        try:
            ok = append_turn_episodes(
                memory_cfg,
                persona_stem,
                user_text=turn.text,
                assistant_text=result.text,
                space_id=space_id,
                thread_key=thread_key,
                message_id=turn_id,
                timestamp=ts,
                peer=turn.author_id,
            )
        except Exception:
            _log.warning(
                "[persona_turn] episode append failed persona=%s",
                persona_stem,
                exc_info=True,
            )
            return
        if not ok:
            _log.warning(
                "[persona_turn] episode append returned false persona=%s",
                persona_stem,
            )
