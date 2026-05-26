from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_DEFAULT_TIMEZONE = "Asia/Tokyo"


def _parse_hhmm(s: str) -> tuple[int, int]:
    parts = s.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"HH:MM 形式ではありません: {s!r}")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"時刻が範囲外です: {s!r}")
    return h, m


def _to_minutes_since_midnight(h: int, m: int) -> int:
    return h * 60 + m


@dataclass
class ScheduleJob:
    id: str
    mode: str  # scheduled | fuzzy_window | interval | chained
    action: str
    notify: str
    timezone: str = _DEFAULT_TIMEZONE
    enabled: bool = True
    action_args: dict[str, Any] = field(default_factory=dict)
    every_day_at: Optional[str] = None
    interval_minutes: Optional[int] = None  # interval モード用（分単位）
    window_start: Optional[str] = None
    window_end: Optional[str] = None
    fuzzy_method: str = "hash"  # hash | random
    on_window_missed: str = "notify"  # notify | silent | mark_done
    slack_channel: Optional[str] = None
    timeout_seconds: int = 600
    memory: bool = False
    persona: Optional[str] = None
    depends_on: list[str] = field(default_factory=list)
    on_chain_failure: str = "abort_notify"  # abort_notify | silent
    every_week_on: Optional[str] = None  # "monday" | ... | "sunday" (None = daily)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_timezone: str = _DEFAULT_TIMEZONE) -> "ScheduleJob":
        jid = str(raw["id"])
        mode = str(raw["mode"])
        action = str(raw["action"])
        notify = str(raw.get("notify", "silent"))
        tz = str(raw.get("timezone", default_timezone))
        enabled = bool(raw.get("enabled", True))
        args = raw.get("action_args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"job {jid}: action_args は辞書である必要があります")
        timeout = int(args.get("timeout_seconds", raw.get("timeout_seconds", 600)))

        every = raw.get("every_day_at")
        interval_min = raw.get("interval_minutes")
        ws = raw.get("window_start")
        we = raw.get("window_end")
        fuzzy_method = str(raw.get("fuzzy_method", "hash"))
        on_missed = str(raw.get("on_window_missed", "notify"))
        slack_ch = raw.get("slack_channel")
        slack_ch = str(slack_ch).strip() if slack_ch else None

        depends_on_raw = raw.get("depends_on") or []
        if isinstance(depends_on_raw, list):
            depends_on = [str(x) for x in depends_on_raw]
        else:
            depends_on = []
        on_chain_failure = str(raw.get("on_chain_failure", "abort_notify"))

        if mode == "scheduled":
            if not every:
                raise ValueError(f"job {jid}: scheduled には every_day_at が必要です")
            _parse_hhmm(str(every))
        elif mode == "interval":
            if not interval_min or int(interval_min) <= 0:
                raise ValueError(f"job {jid}: interval には interval_minutes > 0 が必要です")
        elif mode == "fuzzy_window":
            if not ws or not we:
                raise ValueError(f"job {jid}: fuzzy_window には window_start/end が必要です")
            sh, sm = _parse_hhmm(str(ws))
            eh, em = _parse_hhmm(str(we))
            s_min = _to_minutes_since_midnight(sh, sm)
            e_min = _to_minutes_since_midnight(eh, em)
            if s_min > e_min:
                raise ValueError(
                    f"job {jid}: 日付またぎのウィンドウは禁止です（window_start > window_end）"
                )
        elif mode == "chained":
            # chained: depends_on 全完了でトリガー。every_day_at はフォールバック用（任意）
            if every:
                _parse_hhmm(str(every))
        else:
            raise ValueError(f"job {jid}: 不明な mode: {mode}")

        if fuzzy_method not in ("hash", "random"):
            raise ValueError(f"job {jid}: fuzzy_method は hash か random です")
        if on_missed not in ("notify", "silent", "mark_done"):
            raise ValueError(f"job {jid}: on_window_missed が不正です")
        if notify not in ("silent", "slack_secretary", "slack_custom"):
            raise ValueError(f"job {jid}: notify が不正です")
        if notify == "slack_custom" and not slack_ch:
            raise ValueError(f"job {jid}: slack_custom には slack_channel が必要です")

        memory = bool(raw.get("memory", False))
        persona = raw.get("persona")
        persona = str(persona).strip() if persona else None

        every_week_on_raw = raw.get("every_week_on")
        every_week_on = str(every_week_on_raw).strip().lower() if every_week_on_raw else None
        if every_week_on and every_week_on not in DAY_NAMES:
            raise ValueError(
                f"job {jid}: every_week_on は {DAY_NAMES} のいずれかである必要があります: {every_week_on!r}"
            )

        return cls(
            id=jid,
            mode=mode,
            action=action,
            notify=notify,
            timezone=tz,
            enabled=enabled,
            action_args=args,
            every_day_at=str(every) if every else None,
            interval_minutes=int(interval_min) if interval_min else None,
            window_start=str(ws) if ws else None,
            window_end=str(we) if we else None,
            fuzzy_method=fuzzy_method,
            on_window_missed=on_missed,
            slack_channel=slack_ch,
            timeout_seconds=timeout,
            memory=memory,
            persona=persona,
            depends_on=depends_on,
            on_chain_failure=on_chain_failure,
            every_week_on=every_week_on,
        )

    def target_hhmm_scheduled(self) -> tuple[int, int]:
        assert self.every_day_at
        return _parse_hhmm(self.every_day_at)

    def window_minutes(self) -> tuple[int, int, int]:
        """戻り値: (start_min, end_min, span_minutes 閉区間)."""
        assert self.window_start and self.window_end
        sh, sm = _parse_hhmm(self.window_start)
        eh, em = _parse_hhmm(self.window_end)
        s_min = _to_minutes_since_midnight(sh, sm)
        e_min = _to_minutes_since_midnight(eh, em)
        span = e_min - s_min + 1
        return s_min, e_min, span


ActionFn = Callable[["ScheduleJob"], tuple[bool, str]]
