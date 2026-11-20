from __future__ import annotations


def fanout_prompt_suffix(permission: str | None) -> str:
    """Prompt suffix instructing the engine to emit a ghdag reply block for fan-out."""
    if permission is not None:
        perm_rule = (
            f"Each child must include `permission: {permission}` "
            "(use this exact value on every child).\n"
        )
        child_example = (
            "  - id: subtask-1\n"
            f"    permission: {permission}\n"
            '    prompt: "Instructions for subtask 1"\n'
            "  - id: subtask-2\n"
            f"    permission: {permission}\n"
            '    prompt: "Instructions for subtask 2"\n'
        )
    else:
        perm_rule = ""
        child_example = (
            "  - id: subtask-1\n"
            '    prompt: "Instructions for subtask 1"\n'
            "  - id: subtask-2\n"
            '    prompt: "Instructions for subtask 2"\n'
        )

    return (
        "\n---\n"
        "If independent parallel subtasks are warranted, end your response with a "
        "```ghdag fenced reply block containing `status: ok` and a `children` list.\n"
        "Each child must have a unique `id` (no `:` in id) and a `prompt` with the "
        "full child instruction. Do not use `command`, `engine`, or `model`.\n"
        "Do not emit legacy YAML fan-out blocks after `---`. Omit the reply block "
        "when no parallel subtasks are needed.\n"
        f"{perm_rule}\n"
        "Example:\n\n"
        "```ghdag\n"
        "status: ok\n"
        "children:\n"
        f"{child_example}"
        "```\n"
    )
