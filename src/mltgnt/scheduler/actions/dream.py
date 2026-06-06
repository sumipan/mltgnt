"""mltgnt.scheduler.actions.dream — memory_dream スケジュールアクション。"""
from __future__ import annotations

from pathlib import Path

from mltgnt.config import MemoryConfig
from mltgnt.memory._format import parse_jsonl
from mltgnt.memory.dream import DreamSelector, Synthesizer, read_dream, write_dream
from mltgnt.scheduler.models import ScheduleJob

__all__ = ["run_dream_action"]


def run_dream_action(
    job: ScheduleJob,
    *,
    persona_dir: Path,
    memory_config: MemoryConfig,
) -> tuple[bool, str]:
    persona_stem = job.persona or str(job.action_args.get("persona", "")).strip()
    if not persona_stem:
        return False, f"job {job.id}: persona が未指定です"

    dir_name = memory_config.dream_dir_name
    targets = DreamSelector.pick_targets([persona_dir], memory_dir_name=dir_name)
    if persona_dir not in targets:
        return True, f"dream: {persona_stem} は合成対象外（JSONL 更新なし）"

    memory_dir = persona_dir / dir_name
    entries = []
    for jsonl_path in sorted(memory_dir.glob("*.jsonl")):
        entries.extend(parse_jsonl(jsonl_path))
    if not entries:
        return True, f"dream: {persona_stem} に JSONL エントリがありません"

    existing = read_dream(persona_dir, memory_dir_name=dir_name)

    def llm_call(prompt: str) -> str:
        from mltgnt.bridges.llm_adapter import call_llm

        return str(call_llm(prompt, model=memory_config.dream_model))

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
    return True, f"dream: {persona_stem} を合成しました（{len(summary.sections)} セクション）"
