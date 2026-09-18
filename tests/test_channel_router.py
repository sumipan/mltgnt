"""tests/test_channel_router.py

Ported tests for channel_router.py (Issue #284).
detect_nickname 6 cases + find_observers 4 cases + resolve_responding_persona 11 cases.
New API (Issue #3285): resolve_persona / find_observers_in_space / SpacePersonaEntry.
"""
from __future__ import annotations

import warnings

from mltgnt.routing import ChannelPersonaEntry, SpacePersonaEntry
from mltgnt.routing.channel_router import (
    detect_nickname,
    find_observers,
    find_observers_in_space,
    resolve_persona,
    resolve_responding_persona,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

CHANNEL = "C_TEST"

def _make_map(*entries: ChannelPersonaEntry) -> dict[str, list[ChannelPersonaEntry]]:
    return {CHANNEL: list(entries)}


PERSONA_A = ChannelPersonaEntry(name="persona-a", role="primary", nickname="aki")
PERSONA_B = ChannelPersonaEntry(name="persona-b", role="secondary", nickname="yuki")
PERSONA_C = ChannelPersonaEntry(name="persona-c", role="secondary", nickname="haru")

CHANNEL_MAP_MULTI = _make_map(PERSONA_A, PERSONA_B, PERSONA_C)
CHANNEL_MAP_PRIMARY_ONLY = _make_map(PERSONA_A)


# ---------------------------------------------------------------------------
# detect_nickname tests (6 cases)
# ---------------------------------------------------------------------------

def test_detect_nickname_match():
    result = detect_nickname("aki please", [PERSONA_A, PERSONA_B])
    assert result == "persona-a"


def test_detect_nickname_first_wins():
    result = detect_nickname("akiyuki", [PERSONA_A, PERSONA_B])
    assert result == "persona-a"


def test_detect_nickname_no_match():
    result = detect_nickname("hello", [PERSONA_A, PERSONA_B])
    assert result is None


def test_detect_nickname_empty_text():
    result = detect_nickname("", [PERSONA_A])
    assert result is None


def test_detect_nickname_empty_entries():
    result = detect_nickname("aki", [])
    assert result is None


def test_detect_nickname_empty_nickname_entry():
    entry = ChannelPersonaEntry(name="X", role="primary", nickname="")
    result = detect_nickname("something", [entry])
    assert result is None


# ---------------------------------------------------------------------------
# find_observers tests (4 cases)
# ---------------------------------------------------------------------------

def test_find_observers_excludes_responder():
    result = find_observers("C_TEST", "persona-a", CHANNEL_MAP_MULTI)
    assert result == ["persona-b", "persona-c"]


def test_find_observers_none_responder_returns_all():
    result = find_observers("C_TEST", None, CHANNEL_MAP_MULTI)
    assert result == ["persona-a", "persona-b", "persona-c"]


def test_find_observers_unknown_channel_returns_empty():
    result = find_observers("C_UNKNOWN", "persona-a", CHANNEL_MAP_MULTI)
    assert result == []


def test_find_observers_single_responder_returns_empty():
    result = find_observers("C_TEST", "persona-a", CHANNEL_MAP_PRIMARY_ONLY)
    assert result == []


# ---------------------------------------------------------------------------
# resolve_responding_persona tests (AC#1–#10, #12 — 11 cases)
# ---------------------------------------------------------------------------

def test_nickname_overrides_thread_fixed():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-a"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="yuki, look into this",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "persona-b"


def test_thread_fixed_without_nickname():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-a"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="thanks",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "persona-a"


def test_nickname_switch_updates_fixed():
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-b"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="please continue",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "persona-b"


def test_new_thread_nickname():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="haru, please check",
        thread_ts="2000.0000",
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "persona-c"


def test_new_thread_primary_fallback():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="hello",
        thread_ts="2000.0000",
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "persona-a"


def test_unknown_nickname_fallback():
    thread_ts = "3000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-a"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="unknown-nick, please",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map=thread_persona_map,
    )
    assert result == "persona-a"


def test_unknown_nickname_fallback_no_thread():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="unknown-nick, please",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "persona-a"


def test_partial_nickname_match():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="akiyuki talk",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result == "persona-a"


def test_unknown_channel():
    result = resolve_responding_persona(
        channel="C_UNKNOWN",
        text="yuki, look into this",
        thread_ts=None,
        channel_map=CHANNEL_MAP_MULTI,
        thread_persona_map={},
    )
    assert result is None


def test_primary_only_channel_unchanged():
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="hello",
        thread_ts=None,
        channel_map=CHANNEL_MAP_PRIMARY_ONLY,
        thread_persona_map={},
    )
    assert result == "persona-a"


def test_primary_only_channel_with_thread_fixed():
    thread_ts = "4000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-a"}
    result = resolve_responding_persona(
        channel=CHANNEL,
        text="please continue",
        thread_ts=thread_ts,
        channel_map=CHANNEL_MAP_PRIMARY_ONLY,
        thread_persona_map=thread_persona_map,
    )
    assert result == "persona-a"


def test_thread_fixed_persona_not_in_channel_falls_through_to_primary():
    """Ignore a thread-fixed persona that is not in the channel's entries.

    Repro: persona-e belongs only to another channel (task-society) but was
    written into a C_TEST thread via a delegate result, then incorrectly
    kept responding in C_TEST.
    """
    PERSONA_E = ChannelPersonaEntry(name="persona-e", role="primary", nickname="eve")
    persona_e_channel_map = {"C_TASK_SOCIETY": [PERSONA_E]}
    merged_map = {**CHANNEL_MAP_MULTI, **persona_e_channel_map}

    thread_ts = "5000.0000"
    # persona-e was incorrectly recorded on the C_TEST thread
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-e"}

    result = resolve_responding_persona(
        channel=CHANNEL,
        text="please continue",
        thread_ts=thread_ts,
        channel_map=merged_map,
        thread_persona_map=thread_persona_map,
    )
    # persona-e is not in C_TEST entries, so ignore and return primary (persona-a)
    assert result == "persona-a"


# ---------------------------------------------------------------------------
# resolve_persona / find_observers_in_space / SpacePersonaEntry (Issue #3285)
# ---------------------------------------------------------------------------

SPACE = "space-test"
CONV = "conv-1000"


def _space_map(*entries: SpacePersonaEntry) -> dict[str, list[SpacePersonaEntry]]:
    return {SPACE: list(entries)}


SPACE_MAP_MULTI = _space_map(PERSONA_A, PERSONA_B, PERSONA_C)
SPACE_MAP_PRIMARY_ONLY = _space_map(PERSONA_A)


def test_space_persona_entry_alias():
    """AC-7: SpacePersonaEntry is importable and identical to ChannelPersonaEntry."""
    assert SpacePersonaEntry is ChannelPersonaEntry
    entry = SpacePersonaEntry(name="X", role="primary", nickname="x")
    assert isinstance(entry, ChannelPersonaEntry)


def test_resolve_persona_nickname_overrides_pinned():
    """AC-1: nickname takes priority over pinned."""
    result = resolve_persona(
        "yuki, look into this",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "persona-a"},
    )
    assert result == "persona-b"


def test_resolve_persona_pinned_without_nickname():
    """AC-1: without a nickname, return the pinned persona."""
    result = resolve_persona(
        "thanks",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "persona-a"},
    )
    assert result == "persona-a"


def test_resolve_persona_primary_fallback():
    """AC-1: with no nickname or pin, fall back to primary."""
    result = resolve_persona(
        "hello",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={},
    )
    assert result == "persona-a"


def test_resolve_persona_returns_none_when_no_primary():
    """AC-1: return None when there is no primary either."""
    secondary_only = _space_map(PERSONA_B)
    result = resolve_persona(
        "hello",
        space_id=SPACE,
        conversation_id=None,
        persona_map=secondary_only,
        pinned_personas={},
    )
    assert result is None


def test_resolve_persona_unknown_space_returns_none():
    """AC-2: space_id not in persona_map returns None."""
    result = resolve_persona(
        "yuki, look into this",
        space_id="unknown-space",
        conversation_id=None,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={},
    )
    assert result is None


def test_resolve_persona_conversation_none_skips_pinned():
    """AC-3: conversation_id=None skips pinned and falls through to primary."""
    result = resolve_persona(
        "hello",
        space_id=SPACE,
        conversation_id=None,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "persona-b"},
    )
    assert result == "persona-a"


def test_resolve_persona_pinned_not_in_space_falls_to_primary():
    """AC-4: pinned persona not in the space falls back to primary."""
    persona_e = SpacePersonaEntry(name="persona-e", role="primary", nickname="eve")
    persona_map = {**SPACE_MAP_MULTI, "other-space": [persona_e]}
    result = resolve_persona(
        "please continue",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=persona_map,
        pinned_personas={CONV: "persona-e"},
    )
    assert result == "persona-a"


def test_resolve_responding_persona_compat_matches_and_warns():
    """AC-5: legacy API returns the same result and emits one DeprecationWarning."""
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "persona-a"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = resolve_responding_persona(
            channel=CHANNEL,
            text="thanks",
            thread_ts=thread_ts,
            channel_map=CHANNEL_MAP_MULTI,
            thread_persona_map=thread_persona_map,
        )
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1

    new = resolve_persona(
        "thanks",
        space_id=CHANNEL,
        conversation_id=f"{CHANNEL}:{thread_ts}",
        persona_map=CHANNEL_MAP_MULTI,
        pinned_personas=thread_persona_map,
    )
    assert old == new == "persona-a"


def test_resolve_responding_persona_compat_thread_ts_none():
    """AC-5: thread_ts=None is equivalent to conversation_id=None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = resolve_responding_persona(
            channel=CHANNEL,
            text="hello",
            thread_ts=None,
            channel_map=CHANNEL_MAP_MULTI,
            thread_persona_map={},
        )
    new = resolve_persona(
        "hello",
        space_id=CHANNEL,
        conversation_id=None,
        persona_map=CHANNEL_MAP_MULTI,
        pinned_personas={},
    )
    assert old == new == "persona-a"


def test_find_observers_in_space_excludes_responder():
    """AC-6: returns persona names other than the responder."""
    result = find_observers_in_space(SPACE, "persona-a", SPACE_MAP_MULTI)
    assert result == ["persona-b", "persona-c"]


def test_find_observers_in_space_none_responder_returns_all():
    """AC-6: responding_persona=None returns everyone."""
    result = find_observers_in_space(SPACE, None, SPACE_MAP_MULTI)
    assert result == ["persona-a", "persona-b", "persona-c"]


def test_find_observers_compat_warns():
    """Legacy find_observers emits DeprecationWarning and matches the new API."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = find_observers("C_TEST", "persona-a", CHANNEL_MAP_MULTI)
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    assert old == find_observers_in_space("C_TEST", "persona-a", CHANNEL_MAP_MULTI)


def test_new_api_identifiers_have_no_slack_channel_thread_ts():
    """AC-8: new API identifiers must not contain slack / channel / thread_ts."""
    import inspect

    from mltgnt.routing import channel_router as mod

    for name in ("resolve_persona", "find_observers_in_space", "SpacePersonaEntry"):
        assert "slack" not in name.lower()
        assert "channel" not in name.lower()
        assert "thread_ts" not in name.lower()

    resolve_sig = inspect.signature(resolve_persona)
    for param in resolve_sig.parameters:
        assert "channel" not in param
        assert "thread_ts" not in param
        assert "slack" not in param.lower()

    observer_sig = inspect.signature(find_observers_in_space)
    for param in observer_sig.parameters:
        assert "channel" not in param
        assert "thread_ts" not in param

    # Also check public new API names on the module (exclude compat wrappers)
    new_api_names = {"resolve_persona", "find_observers_in_space"}
    for name in new_api_names:
        assert hasattr(mod, name)
