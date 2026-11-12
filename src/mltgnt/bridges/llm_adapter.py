"""mltgnt.bridges.llm_adapter

L2 bridge: thin wrapper around ghdag.llm.call.
Isolates L3 (domain) from direct L0 (ghdag) dependency.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ghdag.llm import LLMResult


def call_llm(
    prompt: str,
    *,
    engine: str = "",
    model: str = "",
    timeout: int = 120,
) -> "LLMResult":
    """Thin wrapper around ghdag.llm.call. L2 isolates the L0 dependency.

    Read the reply from ``.body``: the engine output adapter normalizes
    per-engine stdout formats there. Codex stdout is JSONL, so using
    ``.stdout`` as the reply would leak JSONL into Slack posts and persona
    memory.

    Returns:
        LLMResult. Body is .body (adapter-extracted); success is .ok.
        Use .stdout only when raw stdout is required.
    """
    from ghdag.llm import call

    return call(prompt, engine=engine, model=model, timeout=timeout)
