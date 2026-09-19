"""mltgnt.persona.schema

Schema definition and validation for persona frontmatter.

FM structure:
    spec_version: str  # optional. Persona schema version (e.g. "2.2.0")

    persona:
      name: str          # required. Must match file stem
      aliases: list[str] # optional
      description: str   # optional

    ops:
      slack:
        username: str
        icon_emoji: str
        icon_url: str
      engine: str
      model: str
      skills: list[str]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mltgnt.config import PERSONA_SECTION_ALIASES

# ---------------------------------------------------------------------------
# Known allowed keys
# ---------------------------------------------------------------------------

_KNOWN_PERSONA_KEYS: frozenset[str] = frozenset({"name", "aliases", "description"})

_KNOWN_OPS_KEYS: frozenset[str] = frozenset({"slack", "engine", "model", "skills"})

_KNOWN_OPS_SLACK_KEYS: frozenset[str] = frozenset(
    {"username", "icon_emoji", "icon_url", "channel", "secondary_channels", "nickname"}
)

# Required sections (## <name> must exist in the body)
REQUIRED_SECTIONS: tuple[str, ...] = (
    "Background",
    "Values",
    "Reaction patterns",
    "Tone",
    "Output format",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PersonaFM:
    """Holds parsed frontmatter."""

    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    spec_version: str | None = None

    # ops namespace
    engine: str = ""
    model: str = ""
    skills: list[str] = field(default_factory=list)
    slack_username: str | None = None
    slack_icon_emoji: str | None = None
    slack_icon_url: str | None = None
    slack_channel: str | None = None
    slack_secondary_channels: list[str] = field(default_factory=list)
    slack_nickname: str | None = None

    # Unknown keys (kept for validation)
    unknown_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse functions
# ---------------------------------------------------------------------------


def parse_fm(meta: dict[str, Any], file_stem: str = "") -> PersonaFM:
    """Build PersonaFM from a frontmatter dict."""
    unknown: list[str] = []

    # —— top-level spec_version ——
    _sv_raw = meta.get("spec_version")
    spec_version: str | None = str(_sv_raw).strip() if _sv_raw is not None else None

    # —— new-schema persona: namespace ——
    persona_ns: dict[str, Any] = meta.get("persona") or {}
    if isinstance(persona_ns, dict):
        name = str(persona_ns.get("name") or file_stem)
        aliases_raw = persona_ns.get("aliases") or []
        aliases = list(aliases_raw) if isinstance(aliases_raw, list) else []
        description = str(persona_ns.get("description") or "")
        for k in persona_ns:
            if k not in _KNOWN_PERSONA_KEYS:
                unknown.append(f"persona.{k}")
    else:
        name = file_stem
        aliases = []
        description = ""

    # —— new-schema ops: namespace ——
    ops_ns: dict[str, Any] = meta.get("ops") or {}
    engine: str = ""
    model: str = ""
    skills: list[str] = []
    slack_username: str | None = None
    slack_icon_emoji: str | None = None
    slack_icon_url: str | None = None
    slack_channel: str | None = None
    slack_secondary_channels: list[str] = []
    slack_nickname: str | None = None

    if isinstance(ops_ns, dict):
        engine = _str_or_none(ops_ns.get("engine")) or ""
        model = _str_or_none(ops_ns.get("model")) or ""
        _skills_raw = ops_ns.get("skills")
        skills = list(_skills_raw) if isinstance(_skills_raw, list) else []
        slack_ops = ops_ns.get("slack") or {}
        if isinstance(slack_ops, dict):
            slack_username = _str_or_none(slack_ops.get("username"))
            slack_icon_emoji = _str_or_none(slack_ops.get("icon_emoji"))
            slack_icon_url = _str_or_none(slack_ops.get("icon_url"))
            slack_channel = _str_or_none(slack_ops.get("channel"))
            _sec_ch = slack_ops.get("secondary_channels")
            slack_secondary_channels = list(_sec_ch) if isinstance(_sec_ch, list) else []
            slack_nickname = _str_or_none(slack_ops.get("nickname"))
            for k in slack_ops:
                if k not in _KNOWN_OPS_SLACK_KEYS:
                    unknown.append(f"ops.slack.{k}")
        for k in ops_ns:
            if k not in _KNOWN_OPS_KEYS:
                unknown.append(f"ops.{k}")

    known_top: frozenset[str] = frozenset({"persona", "ops", "spec_version"})
    for k in meta:
        if k not in known_top:
            unknown.append(k)

    if not name:
        name = file_stem

    return PersonaFM(
        name=name,
        aliases=aliases,
        description=description,
        spec_version=spec_version,
        engine=engine,
        model=model,
        skills=skills,
        slack_username=slack_username,
        slack_icon_emoji=slack_icon_emoji,
        slack_icon_url=slack_icon_url,
        slack_channel=slack_channel,
        slack_secondary_channels=slack_secondary_channels,
        slack_nickname=slack_nickname,
        unknown_keys=unknown,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_fm(fm: PersonaFM) -> ValidationResult:
    """Check FM for schema violations and unknown keys."""
    errors: list[str] = []
    warns: list[str] = []

    if not fm.name:
        errors.append("persona.name is not set")

    for k in fm.unknown_keys:
        errors.append(f"Undefined FM key: {k!r} (add it to the schema before use)")

    return ValidationResult(ok=len(errors) == 0, warnings=warns, errors=errors)


def validate_sections(body: str, fm: PersonaFM) -> ValidationResult:
    """Check that the body contains required sections."""
    warns: list[str] = []
    errors: list[str] = []

    for sec in REQUIRED_SECTIONS:
        aliases = [alias for alias, canonical in PERSONA_SECTION_ALIASES.items() if canonical == sec]
        candidates = (sec, *aliases)
        # Allow both numbered and unnumbered canonical or legacy headings.
        import re

        if not any(
            re.search(rf"^##\s+(?:\d+\.\s+)?{re.escape(candidate)}(?:\s|$)", body, re.MULTILINE)
            for candidate in candidates
        ):
            warns.append(f'Required section "{sec}" not found')

    return ValidationResult(ok=len(errors) == 0, warnings=warns, errors=errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_or_none(val: Any) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Engine / command builders
# ---------------------------------------------------------------------------

VALID_ENGINES: frozenset[str] = frozenset({"claude", "gemini", "cursor", "codex"})

SYSTEM_DEFAULT_ENGINE: str = "claude"
SYSTEM_DEFAULT_MODEL: str = ""
