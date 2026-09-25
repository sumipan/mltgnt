"""mltgnt.scheduler.actions.dream — memory_dream schedule action."""
from __future__ import annotations

from pathlib import Path

from mltgnt.config import MemoryConfig
from mltgnt.memory._format import parse_jsonl
from mltgnt.memory.dream import DreamSelector, Synthesizer, read_dream, write_dream
from mltgnt.scheduler.models import ScheduleJob

__all__ = ["run_dream_action"]

_CLAUDE_DEFAULT_DREAM_MODEL = "claude-haiku-4-5-20251001"


def _resolve_dream_llm(memory_config: MemoryConfig) -> tuple[str, str]:
    """Return (engine, model) for the dream synthesis call."""
    engine = memory_config.dream_engine.strip() or "claude"
    model = memory_config.dream_model.strip()
    if not model and engine == "claude":
        model = _CLAUDE_DEFAULT_DREAM_MODEL
    return engine, model


def run_dream_action(
    job: ScheduleJob,
    *,
    persona_dir: Path,
    memory_config: MemoryConfig,
) -> tuple[bool, str]:
    persona_stem = job.persona or str(job.action_args.get("persona", "")).strip()
    if not persona_stem:
        return False, f"job {job.id}: persona is not set"

    dir_name = memory_config.dream_dir_name
    targets = DreamSelector.pick_targets([persona_dir], memory_dir_name=dir_name)
    if persona_dir not in targets:
        return True, f"dream: {persona_stem} is excluded from synthesis (no JSONL update)"

    memory_dir = persona_dir / dir_name
    entries = []
    for jsonl_path in sorted(memory_dir.glob("*.jsonl")):
        entries.extend(parse_jsonl(jsonl_path))
    if not entries:
        return True, f"dream: {persona_stem} has no JSONL entries"

    existing = read_dream(persona_dir, memory_dir_name=dir_name)
    engine, model = _resolve_dream_llm(memory_config)

    def llm_call(prompt: str) -> str:
        from mltgnt.bridges.llm_adapter import call_llm

        # ghdag applies the engine default model only for None; "" is rejected
        # by its allowlist check, so an empty model is forwarded as None.
        return str(call_llm(
            prompt,
            engine=engine,
            model=model or None,  # type: ignore[arg-type]
        ).body)

    try:
        summary = Synthesizer.synthesize(
            entries,
            existing,
            persona=persona_stem,
            llm_call=llm_call,
        )
    except Exception as e:
        return False, f"dream synthesis failed for {persona_stem}: {e}"

    write_dream(persona_dir, summary, memory_dir_name=dir_name)
    return True, f"dream: synthesized {persona_stem} ({len(summary.sections)} sections)"
