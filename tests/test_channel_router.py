"""tests/test_channel_router.py

channel_router.py の移植テスト（Issue #284）。
detect_nickname 6 ケース + find_observers 4 ケース + resolve_responding_persona 11 ケース。
新 API（Issue #3285）: resolve_persona / find_observers_in_space / SpacePersonaEntry。
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


def test_thread_fixed_persona_not_in_channel_falls_through_to_primary():
    """thread_persona_map に記録された persona がそのチャンネルの entries にない場合は無視する。

    再現ケース: 合田が別チャンネル(task-society)専属なのに C_TEST のスレッドに
    delegate 結果として書き込まれ、その後 C_TEST で合田が応答してしまうバグ。
    """
    GODA = ChannelPersonaEntry(name="合田一人", role="primary", nickname="合田")
    goda_channel_map = {"C_TASK_SOCIETY": [GODA]}
    merged_map = {**CHANNEL_MAP_MULTI, **goda_channel_map}

    thread_ts = "5000.0000"
    # 合田が C_TEST スレッドに誤って記録されたシナリオ
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "合田一人"}

    result = resolve_responding_persona(
        channel=CHANNEL,
        text="続きよろしく",
        thread_ts=thread_ts,
        channel_map=merged_map,
        thread_persona_map=thread_persona_map,
    )
    # 合田は C_TEST の entries にいないので無視され、primary（タチコマ）が返るべき
    assert result == "タチコマ"


# ---------------------------------------------------------------------------
# resolve_persona / find_observers_in_space / SpacePersonaEntry（Issue #3285）
# ---------------------------------------------------------------------------

SPACE = "space-test"
CONV = "conv-1000"


def _space_map(*entries: SpacePersonaEntry) -> dict[str, list[SpacePersonaEntry]]:
    return {SPACE: list(entries)}


SPACE_MAP_MULTI = _space_map(TACHIKOMA, LOGICOMA, FUCHIKOMA)
SPACE_MAP_PRIMARY_ONLY = _space_map(TACHIKOMA)


def test_space_persona_entry_alias():
    """AC-7: SpacePersonaEntry が import でき、ChannelPersonaEntry と同一クラス。"""
    assert SpacePersonaEntry is ChannelPersonaEntry
    entry = SpacePersonaEntry(name="X", role="primary", nickname="x")
    assert isinstance(entry, ChannelPersonaEntry)


def test_resolve_persona_nickname_overrides_pinned():
    """AC-1: ニックネームが pinned より優先。"""
    result = resolve_persona(
        "ロジコマ、これ調べて",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "タチコマ"},
    )
    assert result == "ロジコマ"


def test_resolve_persona_pinned_without_nickname():
    """AC-1: ニックネームなしなら pinned を返す。"""
    result = resolve_persona(
        "ありがとう",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "タチコマ"},
    )
    assert result == "タチコマ"


def test_resolve_persona_primary_fallback():
    """AC-1: ニックネーム・固定なしなら primary。"""
    result = resolve_persona(
        "おはよう",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={},
    )
    assert result == "タチコマ"


def test_resolve_persona_returns_none_when_no_primary():
    """AC-1: primary も無い場合は None。"""
    secondary_only = _space_map(LOGICOMA)
    result = resolve_persona(
        "おはよう",
        space_id=SPACE,
        conversation_id=None,
        persona_map=secondary_only,
        pinned_personas={},
    )
    assert result is None


def test_resolve_persona_unknown_space_returns_none():
    """AC-2: persona_map にない space_id は None。"""
    result = resolve_persona(
        "ロジコマ、これ調べて",
        space_id="unknown-space",
        conversation_id=None,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={},
    )
    assert result is None


def test_resolve_persona_conversation_none_skips_pinned():
    """AC-3: conversation_id=None は pinned をスキップし primary へ。"""
    result = resolve_persona(
        "おはよう",
        space_id=SPACE,
        conversation_id=None,
        persona_map=SPACE_MAP_MULTI,
        pinned_personas={CONV: "ロジコマ"},
    )
    assert result == "タチコマ"


def test_resolve_persona_pinned_not_in_space_falls_to_primary():
    """AC-4: pinned が当該 space にいなければ primary へフォールバック。"""
    goda = SpacePersonaEntry(name="合田一人", role="primary", nickname="合田")
    persona_map = {**SPACE_MAP_MULTI, "other-space": [goda]}
    result = resolve_persona(
        "続きよろしく",
        space_id=SPACE,
        conversation_id=CONV,
        persona_map=persona_map,
        pinned_personas={CONV: "合田一人"},
    )
    assert result == "タチコマ"


def test_resolve_responding_persona_compat_matches_and_warns():
    """AC-5: 旧 API が同じ結果を返し DeprecationWarning を 1 回出す。"""
    thread_ts = "1000.0000"
    thread_persona_map = {f"{CHANNEL}:{thread_ts}": "タチコマ"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = resolve_responding_persona(
            channel=CHANNEL,
            text="ありがとう",
            thread_ts=thread_ts,
            channel_map=CHANNEL_MAP_MULTI,
            thread_persona_map=thread_persona_map,
        )
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1

    new = resolve_persona(
        "ありがとう",
        space_id=CHANNEL,
        conversation_id=f"{CHANNEL}:{thread_ts}",
        persona_map=CHANNEL_MAP_MULTI,
        pinned_personas=thread_persona_map,
    )
    assert old == new == "タチコマ"


def test_resolve_responding_persona_compat_thread_ts_none():
    """AC-5: thread_ts=None は conversation_id=None と同等。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = resolve_responding_persona(
            channel=CHANNEL,
            text="おはよう",
            thread_ts=None,
            channel_map=CHANNEL_MAP_MULTI,
            thread_persona_map={},
        )
    new = resolve_persona(
        "おはよう",
        space_id=CHANNEL,
        conversation_id=None,
        persona_map=CHANNEL_MAP_MULTI,
        pinned_personas={},
    )
    assert old == new == "タチコマ"


def test_find_observers_in_space_excludes_responder():
    """AC-6: 応答者以外のペルソナ名リストを返す。"""
    result = find_observers_in_space(SPACE, "タチコマ", SPACE_MAP_MULTI)
    assert result == ["ロジコマ", "フチコマ"]


def test_find_observers_in_space_none_responder_returns_all():
    """AC-6: responding_persona=None なら全員。"""
    result = find_observers_in_space(SPACE, None, SPACE_MAP_MULTI)
    assert result == ["タチコマ", "ロジコマ", "フチコマ"]


def test_find_observers_compat_warns():
    """旧 find_observers は DeprecationWarning を出し新 API と同じ結果。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = find_observers("C_TEST", "タチコマ", CHANNEL_MAP_MULTI)
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 1
    assert old == find_observers_in_space("C_TEST", "タチコマ", CHANNEL_MAP_MULTI)


def test_new_api_identifiers_have_no_slack_channel_thread_ts():
    """AC-8: 新 API 識別子に slack / channel / thread_ts が現れない。"""
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

    # モジュール直下の公開新 API 名も検査（互換ラッパは除外）
    new_api_names = {"resolve_persona", "find_observers_in_space"}
    for name in new_api_names:
        assert hasattr(mod, name)
