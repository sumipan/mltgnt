"""mltgnt.persona.frontmatter

Parse optional YAML frontmatter at the start of persona Markdown.
"""

from __future__ import annotations

from typing import Any

import yaml


def split_yaml_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """If text starts with `---` … `---` YAML frontmatter, strip it and return the body.

    Otherwise meta is {} and body is the original text.
    On YAML parse failure, return (None, body) (caller decides the error).
    """
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    yaml_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    try:
        meta = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        # Return None to signal error (caller converts to PersonaValidationError)
        return None, body  # type: ignore[return-value]
    if not isinstance(meta, dict):
        return {}, text
    return meta, body


def slack_post_kwargs_from_meta(meta: dict[str, Any]) -> dict[str, str]:
    """Build chat.postMessage kwargs from frontmatter `slack:`."""
    slack = meta.get("slack")
    if not isinstance(slack, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("username", "icon_emoji", "icon_url"):
        val = slack.get(key)
        if val is None:
            continue
        s = str(val).strip()
        if s:
            out[key] = s
    return out

