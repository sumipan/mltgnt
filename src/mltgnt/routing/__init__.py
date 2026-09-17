"""
mltgnt.routing — space → persona routing (media-agnostic).

Origin: ChannelPersonaEntry and load_channel_persona_map() from tools/secretary/config.py
OSS split: receive persona_loader as a callable argument.

Design: Issue #118 §3 (T2) / #3285
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from typing import Any, Callable, Literal

from mltgnt.exceptions import ConfigError, DependencyError

_log = logging.getLogger(__name__)

__all__ = [
    "ChannelPersonaEntry",
    "RoutingRule",
    "SpacePersonaEntry",
    "TRIAGE_PROFILE_MAX_CHARS",
    "detect_nickname",
    "evaluate",
    "extract_json_object",
    "extract_triage_section",
    "find_observers",
    "find_observers_in_space",
    "load_channel_persona_map",
    "prepare_profile_for_triage",
    "resolve_persona",
    "resolve_responding_persona",
]


@dataclass
class RoutingRule:
    """Generic routing rule. First rule whose detector returns True wins."""
    name: str
    detector: Callable[[str, dict[str, Any]], bool]
    handler: str


def evaluate(
    rules: list["RoutingRule"],
    instruction: str,
    ctx: dict[str, Any],
) -> "RoutingRule | None":
    """Scan rules in order; return the first whose detector returns True.
    Return None if no rule matches.

    Args:
        rules: Rules to evaluate (order is priority)
        instruction: User utterance text
        ctx: Extra context for detection (e.g. valid_personas, channel_id)

    Returns:
        Matching RoutingRule, or None
    """
    for rule in rules:
        if rule.detector(instruction, ctx):
            return rule
    return None


@dataclass
class SpacePersonaEntry:
    """Persona role within one space."""
    name: str
    role: Literal["primary", "secondary"]
    nickname: str  # call name (use persona.name when slack_nickname is None)


# Backward-compat alias
ChannelPersonaEntry = SpacePersonaEntry


def load_channel_persona_map(
    persona_loader: Callable[[], list],
) -> dict[str, list[SpacePersonaEntry]]:
    """
    Build a space map from the persona objects returned by persona_loader.
    Return {space_id: list[SpacePersonaEntry]}.
    Personas without a space (ops.slack.channel etc. in the definition) are omitted.
    Raise ConfigError when multiple primaries share one space.


    Keys are treated as space_id (routing does not know media-specific meaning).
    That persona defs may read Slack channel fields is a definition-side concern; this signature stays.

    persona_loader: () -> list of persona objects with attributes:
        - name: str
        - fm.slack_channel: str | None
        - fm.slack_secondary_channels: list[str]
        - fm.slack_nickname: str | None
    """
    result: dict[str, list[SpacePersonaEntry]] = {}
    try:
        personas = persona_loader()
    except Exception as e:
        _log.error("load_channel_persona_map: persona_loader failed: %s", e)
        raise DependencyError(f"persona_loader failed: {e}") from e

    for persona in personas:
        try:
            nickname = persona.fm.slack_nickname or persona.name

            # primary spaces
            ch = persona.fm.slack_channel
            if ch:
                if ch not in result:
                    result[ch] = []
                result[ch].append(SpacePersonaEntry(
                    name=persona.name,
                    role="primary",
                    nickname=nickname,
                ))

            # secondary spaces
            for sec_ch in persona.fm.slack_secondary_channels:
                if sec_ch not in result:
                    result[sec_ch] = []
                result[sec_ch].append(SpacePersonaEntry(
                    name=persona.name,
                    role="secondary",
                    nickname=nickname,
                ))
        except Exception as e:
            _log.warning("load_channel_persona_map: skip persona: %s", e)

    # Duplicate primary check
    for ch, entries in result.items():
        primaries = [e.name for e in entries if e.role == "primary"]
        if len(primaries) > 1:
            raise ConfigError(
                f"Channel {ch} has multiple primaries configured: {primaries}"
            )

    return result


from mltgnt.routing.channel_router import (  # noqa: E402
    detect_nickname,
    find_observers,
    find_observers_in_space,
    resolve_persona,
    resolve_responding_persona,
)
from mltgnt.routing.triage import (  # noqa: E402
    TRIAGE_PROFILE_MAX_CHARS,
    extract_json_object,
    extract_triage_section,
    prepare_profile_for_triage,
)
