"""
tests/test_mltgnt_routing.py — mltgnt.routing のユニットテスト（AC-4）

設計: Issue #118 §7 AC-4
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltgnt.routing import ChannelPersonaEntry, load_channel_persona_map


def test_channel_persona_entry_instantiation() -> None:
    """ChannelPersonaEntry がインスタンス化できる。"""
    entry = ChannelPersonaEntry(name="Test", role="primary", nickname="test")
    assert entry.name == "Test"
    assert entry.role == "primary"
    assert entry.nickname == "test"


def test_channel_persona_entry_secondary() -> None:
    """ChannelPersonaEntry secondary role."""
    entry = ChannelPersonaEntry(name="Other", role="secondary", nickname="other_nick")
    assert entry.role == "secondary"


def test_load_channel_persona_map_builds_map() -> None:
    """load_channel_persona_map がチャンネルマップを構築する。"""
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
    """slack_nickname=None の場合 persona.name をニックネームとして使用する。"""
    persona = MagicMock()
    persona.name = "PersonaA"
    persona.fm.slack_channel = "C_A"
    persona.fm.slack_secondary_channels = []
    persona.fm.slack_nickname = None

    result = load_channel_persona_map(lambda: [persona])

    entry = result["C_A"][0]
    assert entry.nickname == "PersonaA"


def test_load_channel_persona_map_primary_duplicate_exits() -> None:
    """同一チャンネルに primary が2つ → sys.exit(1)。"""
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

    with pytest.raises(SystemExit):
        load_channel_persona_map(lambda: [persona_a, persona_b])


def test_load_channel_persona_map_loader_exception_returns_empty() -> None:
    """persona_loader が例外を投げた場合、空マップを返す。"""
    def failing_loader():
        raise RuntimeError("loader failed")

    result = load_channel_persona_map(failing_loader)
    assert result == {}


def test_load_channel_persona_map_no_channel_skipped() -> None:
    """channel が未設定のペルソナはマップに含まれない。"""
    persona = MagicMock()
    persona.name = "PersonaNoChannel"
    persona.fm.slack_channel = None
    persona.fm.slack_secondary_channels = []
    persona.fm.slack_nickname = "nick"

    result = load_channel_persona_map(lambda: [persona])
    assert result == {}
