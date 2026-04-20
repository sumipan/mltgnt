"""mltgnt.persona.runner

ペルソナのコンテキストを含めてプロンプトを LLM に実行し、応答を返す。

公開 API:
    run_persona_prompt(persona_name, prompt, persona_dir, timeout, memory) -> str
"""
from __future__ import annotations

import logging
import subprocess
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
        timeout: subprocess のタイムアウト秒数。デフォルト 120 秒。
        memory: 呼び出し側が読み込んだメモリ文字列（任意）。
                非 None の場合はプロンプト先頭に付加する。

    Returns:
        LLM の stdout 出力（strip 済み）。
        エラー時は "（タイムアウト）" / "（エラー: {code}）" / "（実行失敗: ...）" を返す。

    Raises:
        FileNotFoundError: ペルソナファイルが見つからない場合。
    """
    from mltgnt.persona.loader import load
    from mltgnt.persona.registry import resolve_with_alias

    pdir = persona_dir if persona_dir is not None else Path("agents")
    path = resolve_with_alias(str(persona_name), pdir)
    persona = load(path)

    if memory is not None:
        effective_prompt = f"{memory}\n\n{prompt}"
    else:
        effective_prompt = prompt

    formatted = persona.format_prompt(effective_prompt)
    cmd = persona.build_command(formatted)

    logger.debug("[runner] persona=%r engine=%r cmd[0]=%r", persona_name, persona.fm.engine, cmd[0])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return proc.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.warning("[runner] persona=%r timeout after %ds", persona_name, timeout)
        return "（タイムアウト）"
    except subprocess.CalledProcessError as e:
        logger.warning("[runner] persona=%r exit code %d", persona_name, e.returncode)
        return f"（エラー: {e.returncode}）"
    except OSError as e:
        logger.warning("[runner] persona=%r OSError: %s", persona_name, e)
        return f"（実行失敗: {e}）"
