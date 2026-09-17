"""
tests/test_mltgnt_routing.py — mltgnt.routing  unit tests（AC-4）

Design: Issue #118 §7 AC-4
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltgnt.exceptions import DependencyError
from mltgnt.routing import ChannelPersonaEntry, load_channel_persona_map


def test_channel_persona_entry_instantiation() -> None:
    """ChannelPersonaEntry"""
    entry = ChannelPersonaEntry(name="Test", role="primary", nickname="test")
    assert entry.name == "Test"
    assert entry.role == "primary"
    assert entry.nickname == "test"


def test_channel_persona_entry_secondary() -> None:
    """ChannelPersonaEntry secondary role."""
    entry = ChannelPersonaEntry(name="Other", role="secondary", nickname="other_nick")
    assert entry.role == "secondary"


def test_load_channel_persona_map_builds_map() -> None:
    """load channel persona map builds channel maps."""
    persona_a = MagicMock()
    persona_a.name = "PersonaA"
    persona_a.fm.slack_channel = "C_A"
    persona_a.fm.slack_secondary_channels = []
    persona_a.fm.slack_nickname = "nick_a"

    persona_b = MagicMock()
    persona_b.name = "PersonaB"
    persona_b.fm.slack_channel = "C_B"
    persona_b.fm.slack_secondary_channels = ["C_A"]
    persona_b.fm.slack_nickname = "nick_b"

    def loader():
        return [persona_a, persona_b]

    result = load_channel_persona_map(loader)

    assert "C_A" in result
    assert "C_B" in result
    c_a_names_roles = {(e.name, e.role) for e in result["C_A"]}
    assert ("PersonaA", "primary") in c_a_names_roles
    assert ("PersonaB", "secondary") in c_a_names_roles


def test_load_channel_persona_map_nickname_fallback() -> None:
    """If slack nickname=None, use persona.name as the command line."""
    persona = MagicMock()
    persona.name = "PersonaA"
    persona.fm.slack_channel = "C_A"
    persona.fm.slack_secondary_channels = []
    persona.fm.slack_nickname = None

    result = load_channel_persona_map(lambda: [persona])

    entry = result["C_A"][0]
    assert entry.nickname == "PersonaA"


def test_load_channel_persona_map_primary_duplicate_raises_config_error() -> None:
    """Two primary channels → ConfigError。"""
    persona_a = MagicMock()
    persona_a.name = "PersonaA"
    persona_a.fm.slack_channel = "C_SAME"
    persona_a.fm.slack_secondary_channels = []
    persona_a.fm.slack_nickname = "nick_a"

    persona_b = MagicMock()
    persona_b.name = "PersonaB"
    persona_b.fm.slack_channel = "C_SAME"  # same channel, also primary
    persona_b.fm.slack_secondary_channels = []
    persona_b.fm.slack_nickname = "nick_b"

    from mltgnt.exceptions import ConfigError

    with pytest.raises(ConfigError, match="primar"):
        load_channel_persona_map(lambda: [persona_a, persona_b])


def test_load_channel_persona_map_loader_exception_raises_dependency_error() -> None:
    """If the persona loader throws an exception, send theependencyE or."""
    def failing_loader():
        raise RuntimeError("connection refused")

    with pytest.raises(DependencyError, match="connection refused"):
        load_channel_persona_map(failing_loader)


def test_load_channel_persona_map_no_channel_skipped() -> None:
    """Persona not set by channel is not included in the map."""
    persona = MagicMock()
    persona.name = "PersonaNoChannel"
    persona.fm.slack_channel = None
    persona.fm.slack_secondary_channels = []
    persona.fm.slack_nickname = "nick"

    result = load_channel_persona_map(lambda: [persona])
    assert result == {}
