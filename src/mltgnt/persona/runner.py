"""mltgnt.persona.runner

Run a prompt against an LLM with persona context and return the reply.

Public API:
    run_persona_prompt(persona_name, prompt, persona_dir, timeout, memory) -> str
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_persona_prompt(
    persona_name: str,
    prompt: str,
    persona_dir: Path | None = None,
    timeout: int = 120,
    memory: str | None = None,
) -> str:
    """Run a prompt against an LLM with persona context and return the reply.

    Args:
        persona_name: Persona name (e.g. "persona-a") or alias.
        prompt: Instruction text for the LLM, wrapped with persona body + format_prompt().
        persona_dir: Persona file directory. Path("agents") when None.
        timeout: LLM call timeout seconds. Default 120.
        memory: Optional memory string loaded by the caller.
                When non-None, prepended to the prompt.

    Returns:
        LLM stdout (stripped).
        On error, return "(error: ...)" / "(exec failed: ...)".

    Raises:
        FileNotFoundError: Persona file not found.
    """
    from mltgnt.bridges.llm_adapter import call_llm
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    pdir = persona_dir if persona_dir is not None else Path("agents")
    path = resolve_with_alias(str(persona_name), pdir)
    persona = load(path)

    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL

    effective_prompt = f"{memory}\n\n{prompt}" if memory is not None else prompt
    formatted = persona.format_prompt(effective_prompt)

    logger.debug("[persona.runner] persona=%r engine=%r", persona_name, engine)

    content: str
    try:
        result = call_llm(formatted, engine=engine, model=model, timeout=timeout)
    except Exception as e:
        logger.warning("[persona.runner] persona=%r exception: %s", persona_name, e)
        content = f"(exec failed: {e})"
    else:
        if not result.success:
            stderr = (result.stderr or "").strip()
            logger.warning("[persona.runner] persona=%r ok=False stderr=%s", persona_name, stderr[:200])
            content = f"(error: {stderr[:200]})" if stderr else "(error)"
        else:
            content = (result.body or "").strip()

    return content
