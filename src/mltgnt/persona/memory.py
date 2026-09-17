"""mltgnt.persona.memory — media-agnostic helpers for persona memory (#3318).

Read/write (paths, locks, LLM) are host-injected.
This module only provides pure helpers (formatting, dedupe key conventions).
"""
from __future__ import annotations

TOOL_TRACE_RESULT_MAX_CHARS = 500

# On-disk-compatible source_tag / dedupe prefixes (keep for existing JSONL)
DIRECT_DEDUPE_PREFIX = "direct-chat"
LEGACY_DIRECT_DEDUPE_PREFIX = "slack-direct"
LEGACY_OBSERVE_DEDUPE_PREFIX = "slack-observe"
LEGACY_DIRECT_SOURCE_TAG = "[slack]"
LEGACY_OBSERVE_SOURCE_TAG = "[slack-observe]"


def format_tool_trace_block(tool_trace: list[dict]) -> str:
    """Convert tool_trace into [tool: ...] / [result: ...] lines."""
    lines = []
    for entry in tool_trace:
        tool_name = entry.get("tool", "")
        args = entry.get("args", {})
        result = entry.get("result", "")
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        lines.append(f"[tool: {tool_name}({args_str})]")
        result_str = str(result)
        if len(result_str) > TOOL_TRACE_RESULT_MAX_CHARS:
            result_str = result_str[:TOOL_TRACE_RESULT_MAX_CHARS] + "…(truncated)"
        lines.append(f"[result: {result_str}]")
    return "\n".join(lines)


def observe_dedupe_key(space_id: str, message_id: str, observer_persona_stem: str) -> str:
    """Dedupe key for observation memory (legacy prefix kept)."""
    return f"{LEGACY_OBSERVE_DEDUPE_PREFIX}:{space_id}:{message_id}:{observer_persona_stem}"


def direct_dedupe_base(space_id: str, thread_key: str, message_id: str) -> str:
    """Dedupe base for direct-reply memory (legacy prefix kept)."""
    return f"{LEGACY_DIRECT_DEDUPE_PREFIX}:{space_id}:{thread_key}:{message_id}"
