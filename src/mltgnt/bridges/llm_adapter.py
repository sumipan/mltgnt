"""mltgnt.bridges.llm_adapter

L2 ブリッジ層: ghdag.llm.call の薄いラッパ。
L3（domain）→ L0（ghdag）の直接依存を隔離する。
"""
from __future__ import annotations


def call_llm(
    prompt: str,
    *,
    engine: str = "",
    model: str = "",
    timeout: int = 120,
):
    """ghdag.llm.call の薄いラッパ。L2 として L0 依存を隔離する。"""
    from ghdag.llm import call

    return call(prompt, engine=engine, model=model, timeout=timeout)
