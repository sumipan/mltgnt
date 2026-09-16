"""
mltgnt.routing — space → ペルソナルーティング（媒体非依存）。

元コード: tools/secretary/config.py の ChannelPersonaEntry と load_channel_persona_map()
OSS 分離: persona_loader を callable 引数で受け取る。

設計: Issue #118 §3 (T2) / #3285
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
    """汎用ルーティングルール。detector が True を返したルールが最初に採用される。"""
    name: str
    detector: Callable[[str, dict[str, Any]], bool]
    handler: str


def evaluate(
    rules: list["RoutingRule"],
    instruction: str,
    ctx: dict[str, Any],
) -> "RoutingRule | None":
    """rules を順に走査し、最初に detector が True を返したルールを返す。
    どのルールにもマッチしなければ None を返す。

    Args:
        rules: 評価するルールのリスト（順序が優先度）
        instruction: ユーザー発話テキスト
        ctx: 検出に必要な追加コンテキスト（例: valid_personas, channel_id）

    Returns:
        マッチした RoutingRule、またはマッチなしなら None
    """
    for rule in rules:
        if rule.detector(instruction, ctx):
            return rule
    return None


@dataclass
class SpacePersonaEntry:
    """1 スペース内でのペルソナの役割を表す。"""
    name: str
    role: Literal["primary", "secondary"]
    nickname: str  # 呼び出し用（slack_nickname が None の場合は persona.name を使う）


# 後方互換別名
ChannelPersonaEntry = SpacePersonaEntry


def load_channel_persona_map(
    persona_loader: Callable[[], list],
) -> dict[str, list[SpacePersonaEntry]]:
    """
    persona_loader が返すペルソナオブジェクトのリストから
    space マップを構築する。
    {space_id: list[SpacePersonaEntry]} の dict を返す。
    space（persona 定義上は ops.slack.channel 等）が未設定のペルソナはマップに含まれない。
    同一 space に primary が複数ある場合は ConfigError を送出する。

    キーは space_id として扱う（routing は媒体固有の意味を知らない）。
    persona 定義が Slack の channel フィールドを読むのは定義側の都合であり、本関数の署名は変えない。

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

            # primary スペース
            ch = persona.fm.slack_channel
            if ch:
                if ch not in result:
                    result[ch] = []
                result[ch].append(SpacePersonaEntry(
                    name=persona.name,
                    role="primary",
                    nickname=nickname,
                ))

            # secondary スペース群
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

    # primary 重複チェック
    for ch, entries in result.items():
        primaries = [e.name for e in entries if e.role == "primary"]
        if len(primaries) > 1:
            raise ConfigError(
                f"チャンネル {ch} に primary が複数設定されています: {primaries}"
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
