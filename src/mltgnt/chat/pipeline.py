"""mltgnt.chat.pipeline

ChatConfig ベースのペルソナ応答パイプライン。

設計方針: diary 固有の設定値（ペルソナ名・ディレクトリ等）は含まず、
呼び出し側が ChatConfig に注入する。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from mltgnt.chat.models import ChatInput, ChatOutput

TZ = ZoneInfo("Asia/Tokyo")

SYSTEM_PROMPT = """\
以下はファイルベースのチャット会話です。
「## user —」で始まるブロックはユーザーの発言、
「## assistant —」で始まるブロックはあなたの過去の応答です。
最後の「## user —」ブロックへの応答を返してください。
応答はMarkdownで書いてください。余計な前置き（「以下が応答です」等）は不要です。"""

CHAT_TIMEOUT_SEC = 300


@dataclass
class ChatConfig:
    """ChatPipeline の設定。diary 固有の値は含まない。"""

    persona_dir: Path
    memory_dir: Path | None = None


class ChatPipeline:
    """ChatConfig を受け取り、ペルソナ応答パイプラインを実行する。"""

    def __init__(self, config: ChatConfig) -> None:
        self._config = config

    def run_pipeline(self, inp: ChatInput) -> ChatOutput:
        """ChatInput を受け取りペルソナ応答を生成して ChatOutput を返す。

        Args:
            inp: チャット入力

        Returns:
            ChatOutput

        Raises:
            FileNotFoundError: 指定ペルソナが見つからない場合
        """
        from mltgnt.persona import load_persona

        persona_stem = inp.persona_name or ""
        if not persona_stem:
            raise ValueError("persona_name が指定されていません")

        persona = load_persona(persona_stem, persona_dir=self._config.persona_dir)

        # context_files（メモリファイル等）の内容を読み込んでプロンプトに注入する
        context_parts: list[str] = []
        for cf in inp.context_files:
            try:
                text = cf.read_text(encoding="utf-8").strip()
                if text:
                    context_parts.append(text)
            except OSError:
                pass

        if inp.context_memory_preferences and inp.context_memory_preferences.strip():
            context_parts.insert(
                0,
                "（ユーザーの好み・傾向）\n\n" + inp.context_memory_preferences.strip(),
            )
        if inp.context_memory_excerpt and inp.context_memory_excerpt.strip():
            context_parts.append(
                "（メモリファイル末尾抜粋・巨大ファイル対策）\n\n"
                + inp.context_memory_excerpt.strip()
            )

        convo_text = _messages_to_text(inp.messages)
        if context_parts:
            context_block = "\n\n".join(context_parts)
            base_task = (
                f"{SYSTEM_PROMPT}\n\n---\n\n"
                f"## このペルソナとの過去の会話記録\n\n{context_block}\n\n---\n\n"
                f"## 現在の会話\n\n{convo_text}"
            )
        else:
            base_task = f"{SYSTEM_PROMPT}\n\n---\n\n{convo_text}"

        prompt = persona.format_prompt(base_task)
        cmd = persona.build_command(prompt)
        content = _run_command(cmd, timeout=CHAT_TIMEOUT_SEC)

        return ChatOutput(
            content=content,
            persona_name=persona_stem,
            timestamp=datetime.now(TZ),
            session_key=inp.session_key,
        )


def _messages_to_text(messages: list[dict]) -> str:
    """messages リストを会話テキスト（チャットファイル形式）に変換する。"""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"## {role} —\n\n{content}")
    return "\n\n---\n\n".join(parts)


def _run_command(cmd: list[str], timeout: int = CHAT_TIMEOUT_SEC) -> str:
    """コマンドリストを実行し出力を返す。"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"コマンドが {timeout}s でタイムアウトしました: {cmd[0]}") from e
    if result.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} が {result.returncode} で終了しました: {result.stderr}"
        )
    return result.stdout.strip()
