"""mltgnt.persona — persona management module

Public API:
    load_persona(name, *, persona_dir)                        -> Persona
    list_personas(persona_dir)                                -> list[str]
    validate_persona(persona, *, available_skills)            -> list[str]
    run_persona_prompt(persona_name, prompt, persona_dir, ..) -> str
    compress_heavy_to_light(heavy_text, *, engine, model, ..) -> str
    regenerate_light_block(persona_path, *, engine, model, ..) -> RegenerationResult
    PersonaValidationError                                    (exception class)

#3318 additions (layer submodules):
    extractor / formatter / memory / phrases / resolve / types / result_format
"""

from __future__ import annotations

from pathlib import Path

from mltgnt.config import PersonaConfig
from mltgnt.persona.compress import compress_heavy_to_light, regenerate_light_block
from mltgnt.persona.formatter import format_persona_body
from mltgnt.persona.loader import Persona, load
from mltgnt.persona.registry import list_personas as _list_personas
from mltgnt.persona.registry import resolve_with_alias
from mltgnt.persona.result_format import format_result_for_persona
from mltgnt.persona.runner import run_persona_prompt
from mltgnt.persona.types import PersonaContext

__all__ = [
    "Persona",
    "PersonaContext",
    "PersonaValidationError",
    "format_persona_body",
    "format_result_for_persona",
    "load_persona",
    "list_personas",
    "validate_persona",
    "run_persona_prompt",
    "compress_heavy_to_light",
    "regenerate_light_block",
]


# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


class PersonaValidationError(Exception):
    """Raised when a persona definition is invalid."""
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_persona(
    name: str,
    *,
    persona_dir: Path | None = None,
    config: PersonaConfig | None = None,
) -> Persona:
    """Load a persona by name or alias.

    Args:
        name: Persona name or alias
        persona_dir: Persona file directory (default: ./agents/)
        config: Persona load settings (defaults when omitted)

    Returns:
        Persona dataclass

    Raises:
        FileNotFoundError: Persona not found
        PersonaValidationError: Invalid frontmatter
    """
    pdir = persona_dir if persona_dir is not None else Path("agents")
    path = resolve_with_alias(name, pdir)
    return load(path, config=config)


def list_personas(persona_dir: Path | None = None) -> list[str]:
    """Return available persona names.

    Args:
        persona_dir: Persona file directory (default: ./agents/)

    Returns:
        List of persona names (stems), sorted
    """
    pdir = persona_dir if persona_dir is not None else Path("agents")
    return _list_personas(pdir)


def validate_persona(
    persona: Persona,
    *,
    available_skills: list[str] | None = None,
) -> list[str]:
    """Validate a persona definition; return warning messages (empty = OK).

    Args:
        persona: Persona object to validate
        available_skills: Available skill name list.
                          Skip skill checks when None.

    Returns:
        List of warning messages (empty = OK)
    """
    messages: list[str] = []

    # Check persona.name matches file stem
    if persona.fm.name and persona.path.stem != persona.fm.name:
        messages.append(
            f"persona.name ({persona.fm.name!r}) does not match filename ({persona.path.stem!r})"
        )

    # Warn on unknown FM keys
    for k in persona.fm.unknown_keys:
        messages.append(f"Undefined FM key: {k!r}")

    # Skill check (only when available_skills is provided)
    if available_skills is not None:
        available_set = set(available_skills)
        for skill in persona.fm.skills:
            if skill not in available_set:
                messages.append(f"Undefined skill: {skill!r}")

    return messages
