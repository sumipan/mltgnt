"""mltgnt.persona.loader

Load and interpret agent files.

Only load() is public. Callers use mltgnt.persona.load_persona().
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import yaml

from mltgnt.bridges.files_adapter import md_read
from mltgnt.config import DEFAULT_WEIGHT_MAP, PERSONA_SECTION_ALIASES, PersonaConfig
from mltgnt.persona.schema import PersonaFM, ValidationResult, parse_fm, validate_fm

PromptFilter = Callable[[str, dict[str, Any]], str]

_TZ = ZoneInfo("Asia/Tokyo")

logger = logging.getLogger(__name__)


def _default_datetime_filter(accumulated: str, ctx: dict[str, Any]) -> str:
    now: datetime = ctx.get("now") or datetime.now(_TZ)
    return accumulated + f"Current datetime: {now.strftime('%Y-%m-%d %H:%M:%S')} (JST)\n\n"


# ---------------------------------------------------------------------------
# Persona object
# ---------------------------------------------------------------------------


@dataclass
class Persona:
    """Holds the contents of an agent file.

    Attributes:
        name:        Persona name (FM persona.name / file stem)
        fm:          Parsed PersonaFM
        sections:    Body section dictionary (for example, "Background" to text)
        body:        Full body without FM
        path:        Source file path
    """

    name: str
    fm: PersonaFM
    sections: dict[str, str]
    body: str
    path: Path
    weight_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_WEIGHT_MAP))
    _prompt_filters: list[tuple[str, PromptFilter]] = field(default_factory=list, init=False, repr=False)

    DEFAULT_OP_MODE: str = "critique"

    def __post_init__(self) -> None:
        self._prompt_filters = [("datetime", _default_datetime_filter)]

    def register_prompt_filter(self, name: str, fn: PromptFilter) -> None:
        self._prompt_filters = [(n, f) for n, f in self._prompt_filters if n != name]
        self._prompt_filters.append((name, fn))

    def format_prompt(self, instruction: str, *, weight: str = "heavy") -> str:
        ctx: dict[str, Any] = {"now": datetime.now(_TZ), "persona": self}
        prefix = ""
        for _, fn in self._prompt_filters:
            prefix = fn(prefix, ctx)

        def _weight_for(key: str) -> str | None:
            """Return section weight via weight_map prefix match. None if no match."""
            for wk, wv in self.weight_map.items():
                if key == wk or key.startswith(wk):
                    return wv
            return None

        # Warn + fallback when sections are missing from weight_map
        unknown = [k for k in self.sections if _weight_for(k) is None]
        if unknown:
            logger.warning(
                "[persona] %r: sections not in weight_map %s — embedding all sections",
                self.name,
                unknown,
            )
            body_part = self.body
        else:
            selected = [f"## {key}\n\n{text}" for key, text in self.sections.items() if _weight_for(key) == weight]
            body_part = "\n\n".join(selected)

        return (
            "You are the following character; reply in their tone and personality.\n\n"
            f"{prefix}"
            f"{body_part}\n\n"
            "--- User instruction ---\n\n"
            f"{instruction}"
        )

    def extract_output_format(self, op_mode: str | None = None) -> str | None:
        """Return the H4 block for op_mode from the output-format section."""
        op_mode = op_mode or self.DEFAULT_OP_MODE
        section = self.sections.get("Output format")
        if section is None:
            return None
        blocks = re.split(r"^#### ", section, flags=re.MULTILINE)
        for block in blocks:
            if block.startswith(op_mode):
                text = block[len(op_mode) :].strip()
                return text if text else None
        return None

    def build_review_prompt(self, op_mode: str = "critique") -> str:
        """Return a prompt fragment for the review system."""
        output_fmt = self.extract_output_format(op_mode)
        parts = [self.body]
        if output_fmt:
            parts.append(f"## Output format\n{output_fmt}")
        return "\n\n".join(parts)


def load(path: Path, *, config: PersonaConfig | None = None) -> Persona:
    """Load a Persona from a file path.

    Raise FileNotFoundError if the file is missing.
    Raise PersonaValidationError if YAML frontmatter parse fails.
    """
    from mltgnt.persona import PersonaValidationError

    if not path.exists():
        raise FileNotFoundError(f"Persona file not found: {path}")

    try:
        md = md_read(path.name, repo_root=path.parent)
    except yaml.YAMLError as e:
        raise PersonaValidationError(f"Failed to parse YAML frontmatter: {path}") from e

    meta = md.frontmatter
    if "persona" not in meta:
        raise PersonaValidationError(f"YAML frontmatter missing required key 'persona': {path}")

    body = md.content.strip()
    fm = parse_fm(meta, file_stem=path.stem)

    # FM validation (log errors)
    result: ValidationResult = validate_fm(fm)
    for err in result.errors:
        logger.warning("[persona] %s: %s", path.name, err)

    aliases = config.section_aliases if config else PERSONA_SECTION_ALIASES
    sections = _parse_sections(body, section_aliases=aliases)

    logger.info(
        "[persona] loaded %r (sections: %s, fm_keys: %s)",
        fm.name,
        list(sections.keys()),
        [k for k in meta if meta[k] is not None],
    )

    wm = dict(config.weight_map) if config else dict(DEFAULT_WEIGHT_MAP)
    return Persona(
        name=fm.name or path.stem,
        fm=fm,
        sections=sections,
        body=body,
        path=path,
        weight_map=wm,
    )


_H2_EXPAND_KEYS: tuple[str, ...] = ("Heavy", "Reference")


def _expand_h3_sections(
    section_text: str,
    *,
    section_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Split into a flat dict by H3 (###) subsections.

    First line of each block is the heading name; the rest is body.
    Ignore pre-H3 content before the first H3 (discard if empty).
    """
    result: dict[str, str] = {}
    aliases = PERSONA_SECTION_ALIASES if section_aliases is None else section_aliases
    parts = re.split(r"^###\s+", section_text, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        lines = part.split("\n", 1)
        raw_key = lines[0].strip()
        key = aliases.get(raw_key, raw_key)
        body = lines[1].strip() if len(lines) > 1 else ""
        if key:
            result[key] = body
    return result


def _parse_sections(
    body: str,
    *,
    section_aliases: dict[str, str] | None = None,
) -> dict[str, str]:
    """Split body into sections by ## headings.

    Heading lines themselves are not included in section bodies.
    For numbered headings like "## 1. Background", remove the number prefix.
    Exclude sections starting with "## 0. ..." (§0).

    In v2, further expand ``## Heavy`` and ``## Reference`` by H3 (###)
    into a flat dict keyed by H3 heading names.
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []
    skip_current: bool = False
    aliases = PERSONA_SECTION_ALIASES if section_aliases is None else section_aliases

    for line in body.splitlines():
        m = re.match(r"^##\s+(\d+\.\s+)?(.+)", line)
        if m:
            if current_key is not None and not skip_current:
                sections[current_key] = "\n".join(current_lines).strip()
            num_prefix = m.group(1) or ""
            raw_title = m.group(2).strip()
            # Exclude §0
            if num_prefix.strip() == "0.":
                skip_current = True
                current_key = None
                current_lines = []
                continue
            skip_current = False
            # Strip full-width bracket annotations used by legacy files.
            raw_title = re.sub(
                r"\s*\N{LEFT BLACK LENTICULAR BRACKET}[^\N{RIGHT BLACK LENTICULAR BRACKET}]*\N{RIGHT BLACK LENTICULAR BRACKET}",
                "",
                raw_title,
            ).strip()
            current_key = aliases.get(raw_title, raw_title)
            current_lines = []
        else:
            if not skip_current:
                current_lines.append(line)

    if current_key is not None and not skip_current:
        sections[current_key] = "\n".join(current_lines).strip()

    # Expand the v2 container sections via H3 into a flat dictionary.
    for h2_key in _H2_EXPAND_KEYS:
        if h2_key in sections:
            h3_sections = _expand_h3_sections(
                sections.pop(h2_key),
                section_aliases=aliases,
            )
            sections.update(h3_sections)

    return sections
