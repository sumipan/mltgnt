"""mltgnt.persona.extractor — H2 section parse and light/heavy text extract (#3318)."""

from __future__ import annotations

import logging
from typing import Literal

from mltgnt.config.language import LanguagePack, get_language_pack

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
    pack: LanguagePack | None = None,
) -> str:
    """Return text for mode, following fallback rules."""
    aliases = (pack or get_language_pack()).persona_section_aliases
    if mode == "light":
        if "Light" in sections:
            return sections["Light"]
        if "Background" in sections:
            return sections["Background"]
        # Keep direct callers that have not passed through the loader working.
        for canonical in ("Light", "Background"):
            legacy = next((k for k in sections if aliases.get(k) == canonical), None)
            if legacy is not None:
                return sections[legacy]
        logger.warning(
            "Persona '%s': light/basic-info sections not found. Falling back to full body.",
            name,
        )
        return body[:500]
    if "Heavy" in sections:
        return sections["Heavy"]
    legacy = next((k for k in sections if aliases.get(k) == "Heavy"), None)
    if legacy is not None:
        return sections[legacy]
    return body
