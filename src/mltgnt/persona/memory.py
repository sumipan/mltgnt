"""mltgnt.persona.memory — 人物像メモリの媒体非依存ヘルパ（#3318）。

読み書き本体（パス・ロック・LLM）はホストが注入する。
本モジュールは整形・dedupe キー規約など純関数のみを提供する。
"""
from __future__ import annotations

TOOL_TRACE_RESULT_MAX_CHARS = 500

# オンディスク互換の source_tag / dedupe 接頭辞（既存 JSONL と突合するため維持）
DIRECT_DEDUPE_PREFIX = "direct-chat"
LEGACY_DIRECT_DEDUPE_PREFIX = "slack-direct"
LEGACY_OBSERVE_DEDUPE_PREFIX = "slack-observe"
LEGACY_DIRECT_SOURCE_TAG = "[slack]"
LEGACY_OBSERVE_SOURCE_TAG = "[slack-observe]"


def format_tool_trace_block(tool_trace: list[dict]) -> str:
    """tool_trace を [tool: ...] / [result: ...] 行に変換する。"""
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
    """観測メモリ用 dedupe キー（レガシー接頭辞維持）。"""
    return f"{LEGACY_OBSERVE_DEDUPE_PREFIX}:{space_id}:{message_id}:{observer_persona_stem}"


def direct_dedupe_base(space_id: str, thread_key: str, message_id: str) -> str:
    """直接応答メモリ用 dedupe ベース（レガシー接頭辞維持）。"""
    return f"{LEGACY_DIRECT_DEDUPE_PREFIX}:{space_id}:{thread_key}:{message_id}"
