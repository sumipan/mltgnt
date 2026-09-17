"""mltgnt.bridges.llm_adapter

L2 bridge: thin wrapper around ghdag.llm.call_text.
Isolates L3 (domain) from direct L0 (ghdag) dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghdag.llm import TextResult


def call_llm(
    prompt: str,
    *,
    engine: str = "",
    model: str = "",
    timeout: int = 120,
) -> "TextResult":
    """Thin wrapper around ghdag.llm.call_text. L2 isolates the L0 dependency.

    Prefer call_text over ghdag.llm.call so per-engine stdout formats are
    normalized by the engine output adapter. Codex EngineSpec always starts
    with ``codex exec - --json``, so raw stdout is JSONL; treating
    call()'s LLMResult.stdout as the body would leak JSONL into Slack posts
    and persona memory.

    Returns:
        TextResult. Body is .body (adapter-extracted); success is .success.
        Use .raw.stdout only when raw stdout is required.
    """
    from ghdag.llm import call_text

    return call_text(prompt, engine=engine, model=model, timeout=timeout)
