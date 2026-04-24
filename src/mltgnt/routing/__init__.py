"""
mltgnt.routing — チャンネル→ペルソナルーティング。

元コード: tools/secretary/config.py の ChannelPersonaEntry と load_channel_persona_map()
OSS 分離: persona_loader を callable 引数で受け取る。

設計: Issue #118 §3 (T2)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Literal

__all__ = [
    "ChannelPersonaEntry",
    "detect_nickname",
    "find_observers",
    "load_channel_persona_map",
    "resolve_responding_persona",
    "resolve_skill",
]


@dataclass
class ChannelPersonaEntry:
    """1チャンネル内でのペルソナの役割を表す。"""
    name: str
    role: Literal["primary", "secondary"]
    nickname: str  # 副チャンネル呼び出し用（slack_nickname が None の場合は persona.name を使う）


def load_channel_persona_map(
    persona_loader: Callable[[], list],
) -> dict[str, list[ChannelPersonaEntry]]:
    """
    persona_loader が返すペルソナオブジェクトのリストから
    チャンネルマップを構築する。
    {channel_id: list[ChannelPersonaEntry]} の dict を返す。
    channel が未設定のペルソナはマップに含まれない。
    同一チャンネルに primary が複数ある場合は stderr にエラー出力して sys.exit(1)。

    persona_loader: () -> list of persona objects with attributes:
        - name: str
        - fm.slack_channel: str | None
        - fm.slack_secondary_channels: list[str]
        - fm.slack_nickname: str | None
    """
    result: dict[str, list[ChannelPersonaEntry]] = {}
    try:
        personas = persona_loader()
    except Exception as e:
        print(f"[routing] load_channel_persona_map: persona_loader failed: {e}", file=sys.stderr)
        return result

    for persona in personas:
        try:
            nickname = persona.fm.slack_nickname or persona.name

            # primary チャンネル
            ch = persona.fm.slack_channel
            if ch:
                if ch not in result:
                    result[ch] = []
                result[ch].append(ChannelPersonaEntry(
                    name=persona.name,
                    role="primary",
                    nickname=nickname,
                ))

            # secondary チャンネル群
            for sec_ch in persona.fm.slack_secondary_channels:
                if sec_ch not in result:
                    result[sec_ch] = []
                result[sec_ch].append(ChannelPersonaEntry(
                    name=persona.name,
                    role="secondary",
                    nickname=nickname,
                ))
        except Exception as e:
            print(f"[routing] load_channel_persona_map: skip persona: {e}", file=sys.stderr)

    # primary 重複チェック
    for ch, entries in result.items():
        primaries = [e.name for e in entries if e.role == "primary"]
        if len(primaries) > 1:
            print(
                f"[routing] ERROR: チャンネル {ch} に primary が複数設定されています: {primaries}",
                file=sys.stderr,
            )
            sys.exit(1)

    return result


async def resolve_skill(
    user_input: str,
    skill_paths: list,
    persona_skills: list[str] | None = None,
    entry_file: str = "SKILL.md",
) -> "tuple | None":
    """
    ユーザー入力からスキルを検索・マッチし、(SkillFile, arguments_str) を返す。

    skill_paths が空または存在しない場合は None を返す（エラーにしない）。
    スキルがマッチしない場合も None を返す。

    引数:
        user_input: ユーザーのメッセージ文字列
        skill_paths: スキルディレクトリのリスト（Path または str）
        persona_skills: ペルソナの skills フィールド（None = フィルタなし）
        entry_file: スキルエントリファイル名
    戻り値:
        (SkillFile, arguments_str) または None
    """
    from pathlib import Path as _Path
    from mltgnt.skill.loader import discover, load
    from mltgnt.skill.matcher import match

    paths = [_Path(p) for p in skill_paths]
    skills = discover(paths, entry_file=entry_file)
    if not skills:
        return None

    result = await match(user_input, skills, persona_skills=persona_skills)
    if result is None:
        return None

    meta, arguments = result
    skill_file = load(meta)
    return (skill_file, arguments)

from mltgnt.routing.channel_router import (  # noqa: E402
    detect_nickname,
    find_observers,
    resolve_responding_persona,
)
