"""mltgnt.persona.runner

ペルソナのコンテキストを含めてプロンプトを LLM に実行し、応答を返す。

公開 API:
    run_persona_prompt(persona_name, prompt, persona_dir, timeout, memory, audit_writer) -> str
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def run_persona_prompt(
    persona_name: str,
    prompt: str,
    persona_dir: Path | None = None,
    timeout: int = 120,
    memory: str | None = None,
    audit_writer: Callable[[dict], None] | None = None,
) -> str:
    """ペルソナのコンテキストを含めてプロンプトを LLM に実行し、応答を返す。

    Args:
        persona_name: ペルソナ名（例: "安宅和人"）またはエイリアス。
        prompt: LLM に渡す指示テキスト。ペルソナの body + format_prompt() でラップされる。
        persona_dir: ペルソナファイルのディレクトリ。None の場合は Path("agents")。
        timeout: LLM 呼び出しのタイムアウト秒数。デフォルト 120 秒。
        memory: 呼び出し側が読み込んだメモリ文字列（任意）。
                非 None の場合はプロンプト先頭に付加する。
        audit_writer: LLM 呼び出し結果を受け取るコールバック（任意）。
                      None の場合は記録しない。

    Returns:
        LLM の stdout 出力（strip 済み）。
        エラー時は "（エラー: ...）" / "（実行失敗: ...）" を返す。

    Raises:
        FileNotFoundError: ペルソナファイルが見つからない場合。
    """
    from mltgnt.chat.pipeline import run_pipeline

    pdir = persona_dir if persona_dir is not None else Path("agents")
    out = run_pipeline(
        prompt,
        persona_name,
        pdir,
        timeout=timeout,
        memory=memory,
        audit_writer=audit_writer,
    )
    return out.content
