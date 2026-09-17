"""
mltgnt.skill.runner — variable substitution and prompt composition.

Design: Issue #124 §6.4
"""
from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path

import yaml

from mltgnt.interfaces.persona import PersonaProtocol
from mltgnt.interfaces.types import ChatInput, Message
from mltgnt.skill.models import SkillFile, SkillRunResult

_VAR_PATTERN = re.compile(r"\$(\d+|\w+)")


def write_result_frontmatter(result_path: Path, run_result: SkillRunResult) -> None:
    """Write produces / skill_io frontmatter at the top of a skill_io:v1 result file.

    Noop when skill_io != "v1" or produces is None.
    """
    if run_result.skill_io != "v1" or run_result.produces is None:
        return
    if not result_path.is_file():
        return

    original = result_path.read_text(encoding="utf-8")
    fm = {
        "skill_io": "v1",
        "produces": {
            "content_type": run_result.produces.content_type,
            "status_markers": list(run_result.produces.status_markers),
        },
    }
    fm_text = yaml.safe_dump(fm, allow_unicode=True, sort_keys=False, default_flow_style=False)
    result_path.write_text(f"---\n{fm_text}---\n{original}", encoding="utf-8")


def _substitute(body: str, arguments: str, persona_name: str, skill_dir: str) -> str:
    """Substitute variables in the skill body."""
    args = arguments.split(" ") if arguments else []

    def replacer(m: re.Match) -> str:
        key = m.group(1)
        if key == "ARGUMENTS":
            return arguments
        if key == "PERSONA":
            return persona_name
        if key == "SKILL_DIR":
            return skill_dir
        if key == "NIKKI_ROOT":
            return os.environ.get("NIKKI_ROOT", "")
        if key == "REPO_ROOT":
            return os.environ.get("REPO_ROOT", "")
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
    extra_context: str | None = None,
) -> SkillRunResult:
    """
    Substitute skill-body variables and compose with persona instructions into SkillRunResult.

    Returns:
        SkillRunResult（chat_input / expected_markers / skill_io）。
        chat_input.model: prefer skill.meta.model; else inherit chat_input.model
        chat_input.messages: system prompt (persona + skill body) + original user message
    """
    skill_dir = str(skill.meta.path.parent.resolve())
    body_substituted = _substitute(
        skill.body,
        arguments,
        persona.name,
        skill_dir,
    )

    if extra_context is not None:
        body_substituted = body_substituted + "\n\n## Context\n\n" + extra_context

    system_content = persona.format_prompt(body_substituted)
    system_message: Message = {"role": "system", "content": system_content}

    # Keep only non-system messages from the original input
    user_messages = [msg for msg in chat_input.messages if msg["role"] != "system"]
    new_messages: list[Message] = [system_message] + user_messages

    new_input = deepcopy(chat_input)
    new_input.messages = new_messages
    new_input.model = skill.meta.model if skill.meta.model is not None else chat_input.model

    expected_markers = (
        list(skill.meta.produces.status_markers)
        if skill.meta.produces
        else []
    )
    return SkillRunResult(
        chat_input=new_input,
        expected_markers=expected_markers,
        skill_io=skill.meta.skill_io,
        produces=skill.meta.produces,
    )
