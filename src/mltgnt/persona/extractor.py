"""mltgnt.persona.extractor — H2 section parse and light/heavy text extract (#3318)."""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def parse_sections(text: str) -> dict[str, str]:
    """Convert to a dict keyed by H2 heading with subordinate text as values.

    Ignore H1 (``# ``). Pass a body with frontmatter already stripped.
    """
    if not text:
        return {}

    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = line[3:].strip()
            current_lines = []
        elif current_key is not None:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def extract(
    sections: dict[str, str],
    mode: Literal["light", "heavy"],
    *,
    body: str = "",
    name: str = "",
) -> str:
    """Return text for mode, following fallback rules."""
    if mode == "light":
        # Japanese text intentionally kept for CJK processing test
        if "軽量" in sections:
            return sections["軽量"]
        if "基本情報" in sections:
            return sections["基本情報"]
        logger.warning(
            "Persona '%s': light/basic-info sections not found."
            " Falling back to full body.",
            name,
        )
        return body[:500]
    # Japanese text intentionally kept for CJK processing test
    if "重量" in sections:
        return sections["重量"]
    return body
