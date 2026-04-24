"""tests/test_channel_router.py

channel_router.py の移植テスト（Issue #284）。
detect_nickname 6 ケース + find_observers 4 ケース + resolve_responding_persona 11 ケース。
"""
from __future__ import annotations

from mltgnt.routing import ChannelPersonaEntry
from mltgnt.routing.channel_router import (
    detect_nickname,
    find_observers,
    resolve_responding_persona,
)

# ---------------------------------------------------------------------------
# テストフィクスチャ
# ---------------------------------------------------------------------------

CHANNEL = "C_TEST"

def _make_map(*entries: ChannelPersonaEntry) -> dict[str, list[ChannelPersonaEntry]]:
    return {CHANNEL: list(entries)}


TACHIKOMA = ChannelPersonaEntry(name="タチコマ", role="primary", nickname="タチコマ")
LOGICOMA  = ChannelPersonaEntry(name="ロジコマ", role="secondary", nickname="ロジコマ")
FUCHIKOMA = ChannelPersonaEntry(name="フチコマ", role="secondary", nickname="フチコマ")

CHANNEL_MAP_MULTI = _make_map(TACHIKOMA, LOGICOMA, FUCHIKOMA)
CHANNEL_MAP_PRIMARY_ONLY = _make_map(TACHIKOMA)


# ---------------------------------------------------------------------------
# detect_nickname のテスト（6 ケース）
# ---------------------------------------------------------------------------

def test_detect_nickname_match():
    result = detect_nickname("タチコマお願い", [TACHIKOMA, LOGICOMA])
    assert result == "タチコマ"


def test_detect_nickname_first_wins():
    result = detect_nickname("タチコマロジコマ", [TACHIKOMA, LOGICOMA])
    assert result == "タチコマ"


def test_detect_nickname_no_match():
    result = detect_nickname("おはよう", [TACHIKOMA, LOGICOMA])
    assert result is None


def test_detect_nickname_empty_text():
    result = detect_nickname("", [TACHIKOMA])
    assert result is None


def test_detect_nickname_empty_entries():
    result = detect_nickname("タチコマ", [])
    assert result is None


def test_detect_nickname_empty_nickname_entry():
    entry = ChannelPersonaEntry(name="X", role="primary", nickname="")
    result = detect_nickname("何か", [entry])
    assert result is None


# ---------------------------------------------------------------------------
# find_observers のテスト（4 ケース）
# ---------------------------------------------------------------------------

def test_find_observers_excludes_responder():
    result = find_observers("C_TEST", "タチコマ", CHANNEL_MAP_MULTI)
    assert result == ["ロジコマ", "フチコマ"]


def test_find_observers_none_responder_returns_all():
    result = find_observers("C_TEST", None, CHANNEL_MAP_MULTI)
    assert result == ["タチコマ", "ロジコマ", "フチコマ"]


def test_find_observers_unknown_channel_returns_empty():
    result = find_observers("C_UNKNOWN", "タチコマ", CHANNEL_MAP_MULTI)
    assert result == []


def test_find_observers_single_responder_returns_empty():
    result = find_observers("C_TEST", "タチコマ", CHANNEL_MAP_PRIMARY_ONLY)
    assert result == []


# ---------------------------------------------------------------------------
# resolve_responding_persona のテスト（AC#1〜#10, #12 の 11 ケース）
# ---------------------------------------------------------------------------

def test_nickname_overrides_thread_fixed():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "タチコマ"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="ロジコマ、これ調べて",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "ロジコマ"


def test_thread_fixed_without_nickname():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "タチコマ"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="ありがとう",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "タチコマ"


def test_nickname_switch_updates_fixed():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "ロジコマ"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="続きお願い",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "ロジコマ"


def test_new_thread_nickname():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="フチコマ、確認して",
        thread_ts="2000.0000",
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "フチコマ"


def test_new_thread_primary_fallback():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="おはよう",
        thread_ts="2000.0000",
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "タチコマ"


def test_unknown_nickname_fallback():
    thread_ts = "3000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "タチコマ"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="ガチコマ、よろしく",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "タチコマ"


def test_unknown_nickname_fallback_no_thread():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="ガチコマ、よろしく",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "タチコマ"


def test_partial_nickname_match():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="タチコマロジコマの話",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "タチコマ"


def test_unknown_channel():
    result = resolve_responding_persona(
        channel="C_UNKNOWN",
        text="ロジコマ、これ調べて",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result is None


def test_primary_only_channel_unchanged():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="おはよう",
        thread_ts=None,
        channel_map=CHANNEL_MAP_PRIMARY_ONLY,
        thread_persona_map={},
    )
    assert result == "タチコマ"


def test_primary_only_channel_with_thread_fixed():
    thread_ts = "4000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "タチコマ"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="続きよろしく",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_PRIMARY_ONLY,
        thread_persona_map=thread_persona_map,
    )
    assert result == "タチコマ"
