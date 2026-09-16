"""src/mltgnt/routing/channel_router.py

マルチスペース（媒体非依存）のエージェントルーティングロジック。

メッセージに対して「誰が応答すべきか」を決定する関数群。
判定優先度: 1. ニックネーム 2. 会話固定 3. primary 4. None

設計: Issue #284 / #3285
"""
from __future__ import annotations

import warnings

from mltgnt.routing import SpacePersonaEntry


def detect_nickname(
    text: str,
    entries: list[SpacePersonaEntry],
) -> str | None:
    """text 内にニックネームが含まれるペルソナ名を返す。複数マッチ時は先勝ち。"""
    for entry in entries:
        if entry.nickname and entry.nickname in text:
            return entry.name
    return None


def resolve_persona(
    text: str,
    *,
    space_id: str,
    conversation_id: str | None,
    persona_map: dict[str, list[SpacePersonaEntry]],
    pinned_personas: dict[str, str],
) -> str | None:
    """メッセージに対して応答すべきペルソナ名を返す。応答不要なら None。

    優先度: ニックネーム → 会話固定（pinned_personas）→ primary → None。
    space_id / conversation_id は不透明文字列（媒体固有の写像は呼び出し側の責務）。
    """
    entries = persona_map.get(space_id)
    if not entries:
        return None

    # 1. ニックネーム検出（会話固定より優先）
    nickname_persona = detect_nickname(text, entries)
    if nickname_persona is not None:
        return nickname_persona

    # 2. 会話固定（space に所属しないペルソナは無視）
    if conversation_id is not None:
        fixed = pinned_personas.get(conversation_id)
        if fixed is not None and any(e.name == fixed for e in entries):
            return fixed

    # 3. primary ペルソナ
    for entry in entries:
        if entry.role == "primary":
            return entry.name

    # 4. None
    return None


def resolve_responding_persona(
    channel: str,
    text: str,
    thread_ts: str | None,
    channel_map: dict[str, list[SpacePersonaEntry]],
    thread_persona_map: dict[str, str],
) -> str | None:
    """互換ラッパ。resolve_persona を呼ぶ。

    .. deprecated::
        Use :func:`resolve_persona` with ``space_id`` / ``conversation_id`` instead.
    """
    warnings.warn(
        "resolve_responding_persona is deprecated; use resolve_persona "
        "(space_id / conversation_id)",
        DeprecationWarning,
        stacklevel=2,
    )
    conversation_id = f"{channel}:{thread_ts}" if thread_ts else None
    return resolve_persona(
        text,
        space_id=channel,
        conversation_id=conversation_id,
        persona_map=channel_map,
        pinned_personas=thread_persona_map,
    )


def find_observers_in_space(
    space_id: str,
    responding_persona: str | None,
    persona_map: dict[str, list[SpacePersonaEntry]],
) -> list[str]:
    """space に参加していて、かつ応答者でないペルソナ名のリストを返す。"""
    entries = persona_map.get(space_id, [])
    observers: list[str] = []
    for entry in entries:
        if entry.name != responding_persona:
            observers.append(entry.name)
    return observers


def find_observers(
    channel: str,
    responding_persona: str | None,
    channel_map: dict[str, list[SpacePersonaEntry]],
) -> list[str]:
    """互換ラッパ。find_observers_in_space を呼ぶ。

    .. deprecated::
        Use :func:`find_observers_in_space` instead.
    """
    warnings.warn(
        "find_observers is deprecated; use find_observers_in_space",
        DeprecationWarning,
        stacklevel=2,
    )
    return find_observers_in_space(channel, responding_persona, channel_map)
