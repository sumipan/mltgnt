"""mltgnt.persona.runner

ペルソナのコンテキストを含めてプロンプトを LLM に実行し、応答を返す。

公開 API:
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
    """ペルソナのコンテキストを含めてプロンプトを LLM に実行し、応答を返す。

    Args:
        persona_name: ペルソナ名（例: "安宅和人"）またはエイリアス。
        prompt: LLM に渡す指示テキスト。ペルソナの body + format_prompt() でラップされる。
        persona_dir: ペルソナファイルのディレクトリ。None の場合は Path("agents")。
        timeout: LLM 呼び出しのタイムアウト秒数。デフォルト 120 秒。
        memory: 呼び出し側が読み込んだメモリ文字列（任意）。
                非 None の場合はプロンプト先頭に付加する。

    Returns:
        LLM の stdout 出力（strip 済み）。
        エラー時は "（エラー: ...）" / "（実行失敗: ...）" を返す。

    Raises:
        FileNotFoundError: ペルソナファイルが見つからない場合。
    """
    from ghdag.llm import call as ghdag_llm_call

    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias
    from mltgnt.persona.schema import SYSTEM_DEFAULT_ENGINE, SYSTEM_DEFAULT_MODEL

    pdir = persona_dir if persona_dir is not None else Path("agents")
    path = resolve_with_alias(str(persona_name), pdir)
    persona = load(path)

    if memory is not None:
        effective_prompt = f"{memory}\n\n{prompt}"
    else:
        effective_prompt = prompt

    formatted = persona.format_prompt(effective_prompt)
    engine = persona.fm.engine or SYSTEM_DEFAULT_ENGINE
    model = persona.fm.model or SYSTEM_DEFAULT_MODEL

    logger.debug("[runner] persona=%r engine=%r", persona_name, engine)

    try:
        result = ghdag_llm_call(formatted, engine=engine, model=model, timeout=timeout)
    except Exception as e:
        logger.warning("[runner] persona=%r exception: %s", persona_name, e)
        return f"（実行失敗: {e}）"

    if not result.ok:
        stderr = (result.stderr or "").strip()
        logger.warning("[runner] persona=%r ok=False stderr=%s", persona_name, stderr[:200])
        return f"（エラー: {stderr[:200]}）" if stderr else "（エラー）"
    return (result.stdout or "").strip()
