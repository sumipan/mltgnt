"""src/mltgnt/routing/channel_router.py

マルチチャンネルエージェントルーティングロジック。

メッセージに対して「誰が応答すべきか」を決定する関数群。
判定優先度: 1. ニックネーム 2. スレッド固定 3. primary 4. None

設計: Issue #284
"""
from __future__ import annotations

from mltgnt.routing import ChannelPersonaEntry


def detect_nickname(
    text: str,
    entries: list[ChannelPersonaEntry],
) -> str | None:
    """text 内にニックネームが含まれるペルソナ名を返す。複数マッチ時は先勝ち。"""
    for entry in entries:
        if entry.nickname and entry.nickname in text:
            return entry.name
    return None


def resolve_responding_persona(
    channel: str,
    text: str,
    thread_ts: str | None,
    channel_map: dict[str, list[ChannelPersonaEntry]],
    thread_persona_map: dict[str, str],
) -> str | None:
    """メッセージに対して応答すべきペルソナ名を返す。応答不要なら None。"""
    entries = channel_map.get(channel)
    if not entries:
        return None

    # 1. ニックネーム検出（スレッド固定より優先）
    nickname_persona = detect_nickname(text, entries)
    if nickname_persona is not None:
        return nickname_persona

    # 2. スレッド固定
    if thread_ts is not None:
        thread_key = f"{channel}:{thread_ts}"
        fixed = thread_persona_map.get(thread_key)
        if fixed is not None:
            return fixed

    # 3. primary ペルソナ
    for entry in entries:
        if entry.role == "primary":
            return entry.name

    # 4. None
    return None


def find_observers(
    channel: str,
    responding_persona: str | None,
    channel_map: dict[str, list[ChannelPersonaEntry]],
) -> list[str]:
    """channel に参加していて、かつ応答者でないペルソナ名のリストを返す。"""
    entries = channel_map.get(channel, [])
    observers: list[str] = []
    for entry in entries:
        if entry.name != responding_persona:
            observers.append(entry.name)
    return observers
