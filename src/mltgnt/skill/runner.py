"""
mltgnt.skill.runner — 変数置換とプロンプト合成。

設計: Issue #124 §6.4
"""
from __future__ import annotations

import re
from copy import deepcopy

from mltgnt.chat.models import ChatInput, Message
from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.skill.models import SkillFile

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
