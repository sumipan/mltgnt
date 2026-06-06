"""mltgnt.memory.dream.synthesizer — LLM による DreamSummary 合成。"""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from mltgnt.config import MemoryConfig
from mltgnt.memory._format import MemoryEntry, assemble_entries_text
from mltgnt.memory.dream._format import DreamSection, DreamSummary
from mltgnt.memory.dream.api import read_dream, read_global
from mltgnt.persona.registry import list_personas

LlmCall = Callable[[str], str]

_DEFAULT_CATEGORIES = ("行動パターン", "好み・傾向")

_SYNTH_PROMPT = """\
以下のメモリエントリを読み、指定カテゴリごとに要約・合成してください。

カテゴリ:
{categories}

出力形式（厳守）:
各カテゴリを `## カテゴリ名` 見出しで区切り、本文のみを記述してください。
前置き・後書き・メタ情報は禁止です。

{existing_block}
【メモリエントリ】
{entries_text}
"""

_GLOBAL_SYNTH_PROMPT = """\
以下は複数ペルソナの記憶要約です。全ペルソナを横断して、ユーザー全体の統合プロファイルを生成してください。

カテゴリ:
{categories}

出力形式（厳守）:
各カテゴリを `## カテゴリ名` 見出しで区切り、本文のみを記述してください。
ペルソナ間で共通するパターンは統合し、矛盾する特徴は文脈を添えて並記してください。
前置き・後書き・メタ情報は禁止です。

{existing_block}{persona_blocks}"""

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*\n(.*?)(?=^##\s+|\Z)", re.MULTILINE | re.DOTALL)

__all__ = ["Synthesizer", "LlmCall"]


class Synthesizer:
    @staticmethod
    def synthesize(
        entries: list[MemoryEntry],
        existing: DreamSummary | None,
        *,
        persona: str,
        llm_call: LlmCall,
        categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
    ) -> DreamSummary:
        if not entries:
            raise ValueError("synthesize requires at least one memory entry")

        entries_text = assemble_entries_text(entries).strip()
        if existing and existing.sections:
            existing_lines = "\n".join(
                f"## {s.category}\n{s.content.strip()}" for s in existing.sections
            )
            existing_block = f"【既存の dream サマリ】\n{existing_lines}\n\n"
        else:
            existing_block = ""

        prompt = _SYNTH_PROMPT.format(
            categories="\n".join(f"- {c}" for c in categories),
            existing_block=existing_block,
            entries_text=entries_text,
        )
        raw = llm_call(prompt)
        parsed = _parse_sections(raw, source_entries=len(entries))
        merged = _merge_sections(existing, parsed)
        updated_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
        return DreamSummary(persona=persona, sections=merged, updated_at=updated_at)

    @staticmethod
    def synthesize_global(
        config: MemoryConfig,
        *,
        llm_call: LlmCall,
        categories: tuple[str, ...] = _DEFAULT_CATEGORIES,
    ) -> DreamSummary:
        exclude = set(config.global_dream_exclude_personas)
        persona_summaries: list[DreamSummary] = []
        for stem in list_personas(config.chat_dir):
            if stem in exclude:
                continue
            dream = read_dream(
                config.chat_dir / stem,
                memory_dir_name=config.dream_dir_name,
            )
            if dream is not None and dream.sections:
                persona_summaries.append(dream)

        if not persona_summaries:
            raise ValueError("no persona dream summaries available for global synthesis")

        existing = read_global(config.chat_dir, memory_dir_name=config.dream_dir_name)

        if existing and existing.sections:
            existing_lines = "\n".join(
                f"## {s.category}\n{s.content.strip()}" for s in existing.sections
            )
            existing_block = f"【既存の global サマリ】\n{existing_lines}\n\n"
        else:
            existing_block = ""

        persona_blocks_parts: list[str] = []
        for summary in persona_summaries:
            block_lines = [f"【ペルソナ: {summary.persona}】"]
            for section in summary.sections:
                block_lines.append(f"## {section.category}")
                block_lines.append(section.content.strip())
            persona_blocks_parts.append("\n".join(block_lines))

        persona_blocks = "\n\n".join(persona_blocks_parts)
        if persona_blocks:
            persona_blocks += "\n"

        prompt = _GLOBAL_SYNTH_PROMPT.format(
            categories="\n".join(f"- {c}" for c in categories),
            existing_block=existing_block,
            persona_blocks=persona_blocks,
        )
        raw = llm_call(prompt)
        parsed = _parse_sections(raw, source_entries=len(persona_summaries))
        merged = _merge_sections(existing, parsed)
        updated_at = datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds")
        return DreamSummary(persona="__global__", sections=merged, updated_at=updated_at)


def _parse_sections(raw: str, *, source_entries: int) -> list[DreamSection]:
    sections: list[DreamSection] = []
    for match in _SECTION_RE.finditer(raw.strip()):
        category = match.group(1).strip()
        content = match.group(2).strip()
        if category and content:
            sections.append(DreamSection(
                category=category,
                content=content,
                source_entries=source_entries,
            ))
    if not sections:
        raise ValueError("LLM response contained no parseable dream sections")
    return sections


def _merge_sections(
    existing: DreamSummary | None,
    new_sections: list[DreamSection],
) -> list[DreamSection]:
    if existing is None:
        return list(new_sections)
    by_category = {s.category: s for s in existing.sections}
    for section in new_sections:
        by_category[section.category] = section
    return list(by_category.values())
