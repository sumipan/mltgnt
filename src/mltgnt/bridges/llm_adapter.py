"""mltgnt.bridges.llm_adapter

L2 ブリッジ層: ghdag.llm.call_text の薄いラッパ。
L3（domain）→ L0（ghdag）の直接依存を隔離する。
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
    """ghdag.llm.call_text の薄いラッパ。L2 として L0 依存を隔離する。

    ghdag.llm.call ではなく call_text を使うのは、エンジンごとの stdout 形式を
    engine output adapter で正規化させるため。codex は EngineSpec が常に
    `codex exec - --json` で起動するため raw stdout は JSONL であり、
    call() の LLMResult.stdout をそのまま本文として扱うと JSONL が
    Slack 投稿やペルソナメモリに漏れる。

    Returns:
        TextResult。本文は .body（adapter 抽出済み）、成否は .success。
        raw stdout が必要な場合のみ .raw.stdout を参照すること。
    """
    from ghdag.llm import call_text

    return call_text(prompt, engine=engine, model=model, timeout=timeout)
