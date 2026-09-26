"""Automatic compaction of conversation logs (#3317).

Keyed by conversation ID. LLM calls are injected callbacks. No external engine SDK dependency.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from mltgnt.conversation.session_store import load_turns, session_path, storage_key

if TYPE_CHECKING:
    from mltgnt.config import ConversationConfig

__all__ = [
    "CompactionResult",
    "audit_path",
    "build_prompt",
    "compact",
    "configure",
    "configure_llm_factory",
    "estimate_tokens",
    "format_turns_for_prompt",
    "load_turns_safe",
    "needs_compaction",
]

_JST = timezone(timedelta(hours=9))

_LlmFactory = Callable[[], Callable[[str], str]]
_llm_factory: _LlmFactory | None = None
_active_config: ConversationConfig | None = None


@dataclass(frozen=True)
class CompactionResult:
    before_turns: int
    after_turns: int
    before_chars: int
    after_chars: int
    summary_chars: int
    personas: list[str] = field(default_factory=list)


def configure(config: ConversationConfig) -> None:
    global _active_config
    _active_config = config


def configure_llm_factory(factory: _LlmFactory | None) -> None:
    """Inject an LLM call factory for summarization."""
    global _llm_factory
    _llm_factory = factory


def estimate_tokens(turns: list[dict]) -> float:
    total_chars = sum(
        len(t.get("content", "") or t.get("summary", ""))
        for t in turns
    )
    return total_chars / 3


_estimate_tokens = estimate_tokens  # deprecated alias


def needs_compaction(
    conversation_id: str,
    *,
    threshold_tokens: int = 16000,
) -> bool:
    turns = load_turns(conversation_id)
    if not turns:
        return False
    return estimate_tokens(turns) > threshold_tokens


def _make_llm_call() -> Callable[[str], str]:
    factory = _llm_factory
    if factory is None:
        raise RuntimeError("session_compact LLM factory is not configured")
    return factory()


def format_turns_for_prompt(
    turns_to_compact: list[dict],
    existing_compacted: list[dict],
) -> str:
    lines: list[str] = []
    for t in existing_compacted:
        content = t.get("content", "")
        lines.append(f"[compacted] {content}")
    for t in turns_to_compact:
        role = t.get("role", "")
        persona = t.get("persona")
        content = t.get("content", "")
        if role == "user":
            lines.append(f"[user] {content}")
        elif role == "assistant":
            tag = f"[assistant:{persona}]" if persona else "[assistant]"
            lines.append(f"{tag} {content}")
        elif role == "bot":
            tag = f"[bot:{persona}]" if persona else "[bot]"
            lines.append(f"{tag} {content}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)


_format_turns_for_prompt = format_turns_for_prompt  # deprecated alias


def build_prompt(turns_to_compact: list[dict], existing_compacted: list[dict]) -> str:
    formatted = format_turns_for_prompt(turns_to_compact, existing_compacted)
    return (
        "Please briefly summarize the following session conversation log.\n\n"
        "- Include each speaker persona name (e.g. persona-a, persona-b) in the summary\n"
        "- Do not omit key topics, decisions, or requests\n"
        "- Output only one paragraph of plain text (no headings, bullets, or metadata)\n\n"
        f"{formatted}"
    )


_build_prompt = build_prompt  # deprecated alias


def audit_path() -> Path:
    if _active_config is not None and _active_config.audit_path is not None:
        return _active_config.audit_path
    if _active_config is not None:
        return _active_config.sessions_dir.parent / "audit.jsonl"
    raise RuntimeError(
        "session_compact is not configured; call mltgnt.conversation.configure() first"
    )


_audit_path = audit_path  # deprecated alias


def compact(
    conversation_id: str,
    *,
    dry_run: bool = False,
    llm_call: Callable[[str], str] | None = None,
) -> CompactionResult:
    call = llm_call or _make_llm_call()

    turns_snapshot = load_turns(conversation_id)
    snapshot_len = len(turns_snapshot)

    turn_entries = [t for t in turns_snapshot if t.get("kind") == "turn"]
    existing_compacted = [t for t in turns_snapshot if t.get("kind") == "compacted"]

    before_turns = len(turn_entries)
    before_chars = sum(len(t.get("content", "")) for t in turn_entries)

    keep_count = max(len(turn_entries) // 3, 5)

    if len(turn_entries) <= 5:
        return CompactionResult(
            before_turns=before_turns,
            after_turns=before_turns,
            before_chars=before_chars,
            after_chars=before_chars,
            summary_chars=0,
            personas=[],
        )

    to_keep = turn_entries[-keep_count:]
    to_compact = turn_entries[:-keep_count]

    personas: list[str] = []
    for t in to_compact:
        p = t.get("persona")
        if p and t.get("role") == "assistant" and p not in personas:
            personas.append(p)
    for c in existing_compacted:
        for p in c.get("personas", []):
            if p not in personas:
                personas.append(p)

    prev_covered = sum(c.get("covered_turns", 0) for c in existing_compacted)
    covered_turns = prev_covered + len(to_compact)

    prompt = build_prompt(to_compact, existing_compacted)
    summary = call(prompt).strip()

    now_ts = datetime.now(_JST).isoformat()
    compacted_entry = {
        "kind": "compacted",
        "ts": now_ts,
        "content": summary,
        "summary": summary,
        "covered_turns": covered_turns,
        "personas": personas,
    }
    summary_chars = len(summary)

    key = storage_key(conversation_id)
    lock_path = session_path(conversation_id).parent / f".{key}.lock"

    if not dry_run:
        with open(lock_path, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                turns_current = load_turns(conversation_id)
                new_turns = turns_current[snapshot_len:]
                final_entries = [compacted_entry] + to_keep + new_turns

                path = session_path(conversation_id)
                tmp = path.parent / f".{path.name}.tmp"
                tmp.write_text(
                    "\n".join(json.dumps(e, ensure_ascii=False) for e in final_entries) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp, path)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    else:
        final_entries = [compacted_entry] + to_keep

    after_turns = len(final_entries)
    after_chars = sum(
        len(e.get("content", "") or e.get("summary", ""))
        for e in final_entries
    )

    result = CompactionResult(
        before_turns=before_turns,
        after_turns=after_turns,
        before_chars=before_chars,
        after_chars=after_chars,
        summary_chars=summary_chars,
        personas=personas,
    )

    if not dry_run:
        record = {
            "source": "session_compact",
            "event_type": "session_compaction",
            "conversation_id": conversation_id,
            "original_turns": before_turns,
            "remaining_turns": after_turns,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "summary_chars": summary_chars,
            "personas": personas,
            "ts": now_ts,
        }
        audit = audit_path()
        audit.parent.mkdir(parents=True, exist_ok=True)
        with open(audit, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return result


def load_turns_safe(
    conversation_id: str,
    *,
    timeout: float = 5.0,
) -> list[dict]:
    key = storage_key(conversation_id)
    lock_path = session_path(conversation_id).parent / f".{key}.lock"
    if not lock_path.exists():
        return load_turns(conversation_id)

    start = time.monotonic()
    with open(lock_path) as lock_fd:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                return load_turns(conversation_id)
            except BlockingIOError:
                if time.monotonic() - start >= timeout:
                    return []
                time.sleep(0.5)
