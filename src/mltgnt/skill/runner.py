"""
mltgnt.skill.runner — 変数置換とプロンプト合成。

設計: Issue #124 §6.4 / Issue #907
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

from mltgnt.chat.models import ChatInput, Message
from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.skill.models import SkillFile

if TYPE_CHECKING:
    from mltgnt.agent._runner import AgentResult, LLMCaller, ToolExecutor

_VAR_PATTERN = re.compile(r"\$(\d+|\w+)")


def _substitute(body: str, arguments: str, persona_name: str, skill_dir: str) -> str:
    """スキル本文の変数を置換する。"""
    args = arguments.split(" ") if arguments else []

    def replacer(m: re.Match) -> str:
        key = m.group(1)
        if key == "ARGUMENTS":
            return arguments
        if key == "PERSONA":
            return persona_name
        if key == "SKILL_DIR":
            return skill_dir
        if key.isdigit():
            idx = int(key)
            return args[idx] if idx < len(args) else ""
        return m.group(0)

    return _VAR_PATTERN.sub(replacer, body)


def run(
    skill: SkillFile,
    persona: PersonaProtocol,
    arguments: str,
    chat_input: ChatInput,
) -> ChatInput:
    """
    スキル本文の変数を置換し、ペルソナ指示と合成した ChatInput を返す。

    戻り値:
        新しい ChatInput。
        - model: skill.meta.model が優先、None なら chat_input.model を引き継ぐ
        - messages: システムプロンプト（ペルソナ指示 + スキル本文）+ 元のユーザーメッセージ
    """
    skill_dir = str(skill.meta.path.parent.resolve())
    body_substituted = _substitute(
        skill.body,
        arguments,
        persona.name,
        skill_dir,
    )

    system_content = persona.format_prompt(body_substituted)
    system_message: Message = {"role": "system", "content": system_content}

    # Keep only non-system messages from the original input
    user_messages = [msg for msg in chat_input.messages if msg["role"] != "system"]
    new_messages: list[Message] = [system_message] + user_messages

    new_input = deepcopy(chat_input)
    new_input.messages = new_messages
    new_input.model = skill.meta.model if skill.meta.model is not None else chat_input.model

    return new_input


def run_agent(
    skill: SkillFile,
    persona: PersonaProtocol,
    arguments: str,
    chat_input: ChatInput,
    llm_call: "LLMCaller",
    tool_executor: "ToolExecutor",
    terminal_tools: frozenset[str],
    max_iterations: int = 3,
) -> "AgentResult | None":
    """skill を AgentRunner ループで実行する。

    skill.meta.agent が True のスキル向け。
    skill.meta.tools に定義されたツール一覧が AgentRunner に渡る。
    agent: false のスキルに対しても呼び出せるが、通常は skill.meta.agent を確認してから使う。
    """
    from mltgnt.agent._runner import AgentRunner

    prepared = run(skill, persona, arguments, chat_input)
    system_prompt = prepared.messages[0]["content"] if prepared.messages else ""
    runner = AgentRunner(
        llm_call=llm_call,
        tool_executor=tool_executor,
        terminal_tools=terminal_tools,
        tools=skill.meta.tools,
        max_iterations=max_iterations,
    )
    return runner.run(system_prompt)
